"""Send changes made in the downloaded source to the developers for review.

After the user approves a change in Codex, Codex runs ``espi-live
--send-changes`` (see ``codex.py``'s guide). That compares the source copy
with its original download (``source_code``) and files a GitHub issue
labelled ``codex-change`` holding the title and description Codex wrote in
``CHANGE_REQUEST.md``, a readable diff and the exact patch (gzip and base64,
so no Markdown or line-ending handling can alter it). The
``codex-change.yml`` workflow reads it back with ``python -m
espi_live.changes extract``, applies the patch to a new branch from the same
commit and opens a pull request. Both ends use this module, so the format
stays in step. The issue goes through the Issues-only feedback token: the
program holds no credential that can write code. No GUI imports.
"""

import argparse
import base64
import binascii
import gzip
import os
import re
import sys
import textwrap
import zlib
from dataclasses import dataclass

from espi_live import feedback, source_code

LABEL = "codex-change"
MAX_BODY = 65000  # GitHub refuses issue bodies over 65,536 characters
MAX_TITLE = 120
DESCRIPTION_END = "<!-- espi-change-description-end -->"
BASE_PATTERN = re.compile(r"<!-- espi-change base=([0-9a-f]{40}) -->")
PATCH_PATTERN = re.compile(r"<!-- espi-change-patch\s+([A-Za-z0-9+/=\s]+?)\s*-->")
GIT_HEADER = re.compile(r"diff --git a/(.+) b/(.+)$")
HUNK_HEADER = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
FILE_MODES = ("new file mode 100644", "deleted file mode 100644")
# The workflow's token cannot push workflow files, and changes to CI are
# for developers to make.
PROTECTED_PREFIXES = (".github/", ".git/")


class ChangeError(Exception):
    """A change that could not be sent, or an issue that holds no usable change."""


@dataclass(frozen=True)
class ChangeRequest:
    """One change, ready to file."""

    title: str
    description: str
    patch: str
    base: str  # full commit SHA the patch applies to
    files: tuple = ()  # (relative path, "added"/"modified"/"deleted")
    skipped: tuple = ()  # binary files that cannot be sent
    earlier: tuple = ()  # issue URLs of changes sent before from this copy


def read_request(folder):
    """Read the title and description Codex wrote for a change.

    Args: folder is the source copy.
    Returns: (title, description). Raises ChangeError when the file is
        missing or has no title.
    """
    path = os.path.join(folder, source_code.REQUEST_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        text = ""
    title, _, description = text.partition("\n")
    title = title.strip().lstrip("#").strip()
    if not title:
        raise ChangeError(f"write {source_code.REQUEST_FILE} in {folder} first: a short "
                          "title on the first line, then a plain-language description")
    return title[:MAX_TITLE], description.strip()


def encode_patch(patch):
    """Pack a patch so it survives an issue body byte for byte.

    Args: patch is the patch text.
    Returns: Base64 of the gzip-compressed UTF-8 patch, in 76-character lines.
    """
    packed = base64.b64encode(gzip.compress(patch.encode("utf-8"), mtime=0)).decode("ascii")
    return "\n".join(textwrap.wrap(packed, 76))


def decode_patch(text):
    """Unpack a patch written by encode_patch.

    Args: text is the base64 text (whitespace is ignored).
    Returns: The patch text. Raises ChangeError if it is damaged.
    """
    try:
        return gzip.decompress(base64.b64decode("".join(text.split()),
                                                validate=True)).decode("utf-8")
    except (binascii.Error, OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
        raise ChangeError(f"the patch in the issue is damaged: {exc}") from exc


def fence(text):
    """Choose a Markdown code fence that the text cannot close.

    Args: text is what goes inside the fence.
    Returns: A run of backticks longer than any run in text (at least 3).
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def issue_body(request, readable=True):
    """Write the issue text for a change.

    Args: request is the ChangeRequest; readable adds the diff for people
        (left out when the body would be too long).
    Returns: The Markdown body.
    """
    files = ", ".join(f"`{path}` ({kind})" for path, kind in request.files)
    lines = [request.description or "_No description given._", "", DESCRIPTION_END, "",
             "---",
             "_Sent from ESPI live's **Open in Codex** button after the user approved this "
             "change in Codex. The `codex-change` workflow turns it into a pull request._", "",
             *feedback.environment_lines(),
             f"- Base commit: `{request.base}`",
             f"- Files: {files}"]
    if request.skipped:
        lines.append("- Not included (binary files cannot be sent): "
                     + ", ".join(f"`{path}`" for path in request.skipped))
    if request.earlier:
        lines.append("- Also contains the changes sent earlier in " + ", ".join(request.earlier))
    if readable:
        marks = fence(request.patch)
        lines += ["", "<details><summary>Patch</summary>", "", f"{marks}diff",
                  request.patch.rstrip("\n"), marks, "", "</details>"]
    lines += ["", f"<!-- espi-change base={request.base} -->",
              "<!-- espi-change-patch", encode_patch(request.patch), "-->"]
    return "\n".join(lines) + "\n"


def issue_payload(request):
    """Build the GitHub "create issue" body for a change.

    Args: request is the ChangeRequest.
    Returns: A dict with title, body and labels. Raises ChangeError when the
        change is too large for an issue.
    """
    body = issue_body(request)
    if len(body) > MAX_BODY:
        body = issue_body(request, readable=False)
    if len(body) > MAX_BODY:
        raise ChangeError(f"the change is too large to send this way ({len(body):,} "
                          f"characters; the limit is {MAX_BODY:,})")
    return {"title": request.title, "body": body, "labels": [LABEL]}


def prepare_request(folder):
    """Collect everything needed to send the changes in a copy.

    Args: folder is the downloaded source copy.
    Returns: (ChangeRequest, digest of its patch, URL of an earlier send of
        exactly this patch or None). Raises ChangeError or SourceError.
    """
    state = source_code.read_state(folder)
    if state is None:
        raise ChangeError(f"{folder} is not a source copy downloaded by ESPI live; send "
                          "changes from the folder that Open in Codex opens")
    changes = source_code.compare(folder)
    patch, skipped = source_code.make_patch(changes)
    if not patch:
        detail = (" (binary files cannot be sent: " + ", ".join(skipped) + ")") if skipped else ""
        raise ChangeError(f"nothing to send: no text file differs from the original{detail}")
    check_patch(patch)  # the workflow would refuse it; say so before sending
    digest = source_code.patch_digest(patch)
    records = source_code.sent_records(folder)
    for record in records:
        if record["patch"] == digest:
            return None, digest, record["url"]
    title, description = read_request(folder)
    files = tuple((path, source_code.change_kind(old, new)) for path, old, new in changes
                  if path not in skipped)
    request = ChangeRequest(title, description, patch, state["commit"], files,
                            tuple(skipped), tuple(record["url"] for record in records))
    return request, digest, None


def send(folder=None):
    """File the changes in the source copy as a GitHub issue.

    Args: folder is the copy; None uses source_code.source_folder().
    Returns: (issue URL, True if that issue already held these changes).
        Raises ChangeError (or source_code.SourceError) on failure.
    """
    folder = folder or source_code.source_folder()
    request, digest, earlier = prepare_request(folder)
    if earlier:
        return earlier, True
    try:
        url = feedback.create_issue(issue_payload(request))
    except feedback.FeedbackError as exc:
        raise ChangeError(str(exc)) from exc
    try:
        source_code.record_sent(folder, url, digest)
        os.remove(os.path.join(folder, source_code.REQUEST_FILE))
    except OSError:
        pass  # sent all the same; at worst it can be sent again
    return url, False


def send_from_cli():
    """Run ``--send-changes``: send the changes and print the outcome.

    Args: None.
    Returns: 0 when sent (or already sent), 1 on failure.
    """
    try:
        url, already = send()
    except (ChangeError, source_code.SourceError) as exc:
        print(f"Could not send the changes: {exc}", file=sys.stderr)
        return 1
    if already:
        print(f"These changes were already sent: {url}")
    else:
        print(f"Sent. The ESPI live developers will review the change here: {url}")
    return 0


def patch_paths(patch):
    """List the files a patch changes, reading it as strictly as make_patch writes it.

    Each hunk is walked by the line counts in its header, so no text inside
    a hunk can pass for another file's header.

    Args: patch is the patch text.
    Returns: A list of paths, without the a/ and b/ prefixes. Raises
        ChangeError if the patch has any other form.
    """
    lines = patch.split("\n")
    if lines[-1] == "":
        lines.pop()
    names, index = [], 0

    def fail(reason):
        """Reject the patch.

        Args: reason says what was wrong.
        Returns: Never; raises ChangeError.
        """
        raise ChangeError(f"the patch is malformed: {reason} at line {index + 1}")

    while index < len(lines):
        header = GIT_HEADER.match(lines[index])
        if not header or header.group(1) != header.group(2):
            fail("expected a file header")
        name = header.group(1)
        names.append(name)
        index += 1
        if index < len(lines) and lines[index] in FILE_MODES:
            index += 1
        if index < len(lines) and lines[index].startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                fail("a --- line without its +++ line")
            if lines[index][4:] not in ("/dev/null", f"a/{name}") or \
                    lines[index + 1][4:] not in ("/dev/null", f"b/{name}"):
                fail("file names that disagree")
            index += 2
        while index < len(lines) and lines[index].startswith("@@"):
            hunk = HUNK_HEADER.match(lines[index])
            if not hunk:
                fail("a damaged hunk header")
            old, new = (1 if count is None else int(count) for count in hunk.groups())
            index += 1
            while old > 0 or new > 0:
                mark = lines[index][:1] if index < len(lines) else ""
                if mark not in (" ", "-", "+", "\\"):
                    fail("a hunk that ends early")
                old -= mark in (" ", "-")
                new -= mark in (" ", "+")
                if old < 0 or new < 0:
                    fail("a hunk longer than its header says")
                index += 1
            if index < len(lines) and lines[index].startswith("\\"):
                index += 1
    return names


def check_patch(patch):
    """Refuse patches that reach outside the project or into protected files.

    Args: patch is the patch text.
    Returns: None. Raises ChangeError naming the first offending path.
    """
    names = patch_paths(patch)
    if not names:
        raise ChangeError("the issue's patch changes no files")
    for name in names:
        parts = name.split("/")
        if name.startswith(("/", '"')) or "\\" in name or ".." in parts or \
                name.startswith(PROTECTED_PREFIXES):
            raise ChangeError(f"the patch may not change {name}")


def parse_issue(body):
    """Read a change back out of an issue body written by issue_body.

    Args: body is the issue text.
    Returns: (base commit, patch text, description). Raises ChangeError
        when the issue holds no usable change.
    """
    base = BASE_PATTERN.search(body or "")
    packed = PATCH_PATTERN.search(body or "")
    if not base or not packed:
        raise ChangeError("this issue holds no change from ESPI live")
    patch = decode_patch(packed.group(1))
    check_patch(patch)
    description = body.partition(DESCRIPTION_END)[0].replace("\r\n", "\n").strip()
    return base.group(1), patch, description


def main(argv=None):
    """Command line for the codex-change workflow.

    ``extract`` reads the issue body from the ISSUE_BODY environment
    variable (never from the command line, so its text cannot reach a
    shell), writes the patch and the pull request's description, and prints
    the base commit.

    Args: argv is the argument list; None uses sys.argv[1:].
    Returns: 0 on success, 1 when the issue holds no usable change (the
        reason goes to stderr).
    """
    parser = argparse.ArgumentParser(prog="python -m espi_live.changes")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="read a codex-change issue")
    extract.add_argument("--patch", required=True, help="file to write the patch to")
    extract.add_argument("--description", required=True,
                         help="file to write the change's description to")
    args = parser.parse_args(argv)
    try:
        base, patch, description = parse_issue(os.environ.get("ISSUE_BODY", ""))
    except ChangeError as exc:
        print(exc, file=sys.stderr)
        return 1
    with open(args.patch, "w", encoding="utf-8", newline="") as handle:
        handle.write(patch)
    with open(args.description, "w", encoding="utf-8") as handle:
        handle.write(description + "\n")
    print(base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
