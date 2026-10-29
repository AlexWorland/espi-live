"""Network-free checks for sending source changes as GitHub issues and for
reading them back in the codex-change workflow.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import base64
import contextlib
import gzip
import io
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import warnings
from unittest import mock

from espi_live import changes, feedback, source_code
from source_fixtures import COMMIT, make_copy, prebuilt, write

PATCH = ("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
         "@@ -1,2 +1,2 @@\n # ESPI live\n-old\n+new\n")


def request(**fields):
    """Build a ChangeRequest with test defaults.

    Args: fields override the defaults.
    Returns: A changes.ChangeRequest.
    """
    values = {"title": "Faster preview", "description": "Draw frames sooner.",
              "patch": PATCH, "base": COMMIT, "files": (("README.md", "modified"),)}
    values.update(fields)
    return changes.ChangeRequest(**values)


class RequestFileTests(unittest.TestCase):
    """Reading CHANGE_REQUEST.md."""

    def setUp(self):
        """Create a temporary copy folder.

        Args: None.
        Returns: None.
        """
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_title_and_description(self):
        """The first line is the title (heading marks dropped, length capped).

        Args: None.
        Returns: None.
        """
        write(self.folder, "CHANGE_REQUEST.md", "# Show the frame rate\n\nIn the corner.\n")
        self.assertEqual(changes.read_request(self.folder),
                         ("Show the frame rate", "In the corner."))
        write(self.folder, "CHANGE_REQUEST.md", "x" * 300)
        self.assertEqual(changes.read_request(self.folder), ("x" * 120, ""))

    def test_missing_or_empty_file_is_refused(self):
        """Without a title the change cannot be sent.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(changes.ChangeError, "write CHANGE_REQUEST.md"):
            changes.read_request(self.folder)
        write(self.folder, "CHANGE_REQUEST.md", "#\n\nonly a body")
        with self.assertRaisesRegex(changes.ChangeError, "write CHANGE_REQUEST.md"):
            changes.read_request(self.folder)


class FormatTests(unittest.TestCase):
    """The issue body and the packed patch."""

    def test_patch_packing_round_trips_exactly(self):
        """CRLF, tabs and non-ASCII survive packing.

        Args: None.
        Returns: None.
        """
        patch = PATCH + "+tab\there\r\n+größe\n"
        packed = changes.encode_patch(patch * 20)
        self.assertTrue(all(len(line) <= 76 for line in packed.splitlines()))
        self.assertEqual(changes.decode_patch(packed), patch * 20)

    def test_damaged_patches_are_refused(self):
        """Bad base64, a bad or truncated gzip stream and non-UTF-8 are refused.

        Args: None.
        Returns: None.
        """
        whole = gzip.compress(b"ok", mtime=0)
        for text in ("not base64!", base64.b64encode(b"not gzip").decode(),
                     base64.b64encode(whole[:-6]).decode(),
                     base64.b64encode(gzip.compress(b"\xff\xfe")).decode()):
            with self.subTest(text=text), \
                    self.assertRaisesRegex(changes.ChangeError, "damaged"):
                changes.decode_patch(text)

    def test_fence_outlasts_backticks_in_the_patch(self):
        """The fence is longer than any backtick run in the text.

        Args: None.
        Returns: None.
        """
        self.assertEqual(changes.fence("no ticks"), "```")
        self.assertEqual(changes.fence("```` and `"), "`````")

    def test_body_holds_description_facts_diff_and_patch(self):
        """The body leads with the description and carries everything else.

        Args: None.
        Returns: None.
        """
        body = changes.issue_body(request(skipped=("logo.png",),
                                          earlier=("https://x/issues/1",)))
        self.assertTrue(body.startswith("Draw frames sooner.\n"))
        self.assertIn(f"- Base commit: `{COMMIT}`", body)
        self.assertIn("- Files: `README.md` (modified)", body)
        self.assertIn("binary files cannot be sent): `logo.png`", body)
        self.assertIn("sent earlier in https://x/issues/1", body)
        self.assertIn("```diff\n" + PATCH.rstrip("\n") + "\n```", body)
        self.assertEqual(changes.parse_issue(body), (COMMIT, PATCH, "Draw frames sooner."))
        edited = changes.issue_body(request(description="Line one\n\nLine two"))
        self.assertEqual(changes.parse_issue(edited.replace("\n", "\r\n")),
                         (COMMIT, PATCH, "Line one\n\nLine two"))
        bare = changes.issue_body(request(description=""), readable=False)
        self.assertIn("_No description given._", bare)
        self.assertNotIn("<details>", bare)

    def test_payload_is_labelled_and_shrinks_to_fit(self):
        """A long change drops the readable diff; a huge one is refused.

        Args: None.
        Returns: None.
        """
        payload = changes.issue_payload(request())
        self.assertEqual(payload["labels"], ["codex-change"])
        self.assertEqual(payload["title"], "Faster preview")
        self.assertIn("<details>", payload["body"])
        compact = len(changes.issue_body(request(), readable=False))
        with mock.patch.object(changes, "MAX_BODY", compact):
            self.assertNotIn("<details>", changes.issue_payload(request())["body"])
        with mock.patch.object(changes, "MAX_BODY", 100), \
                self.assertRaisesRegex(changes.ChangeError, "too large"):
            changes.issue_payload(request())


class PatchCheckTests(unittest.TestCase):
    """patch_paths and check_patch, which guard the workflow."""

    def refused(self, patch, message):
        """Assert that check_patch refuses a patch.

        Args: patch is the patch text; message is a regex for the error.
        Returns: None.
        """
        with self.assertRaisesRegex(changes.ChangeError, message):
            changes.check_patch(patch)

    def test_reads_every_form_make_patch_writes(self):
        """Added, deleted and edited files, and missing final line breaks.

        Args: None.
        Returns: None.
        """
        patch, _ = source_code.make_patch([
            ("a.txt", None, b"new"), ("b.txt", b"gone\n", None),
            ("c.txt", b"one", b"two\nthree"), ("e.txt", None, b"")])
        self.assertEqual(changes.patch_paths(patch), ["a.txt", "b.txt", "c.txt", "e.txt"])
        changes.check_patch(patch)

    def test_malformed_patches_are_refused(self):
        """Anything but make_patch's form is refused, so no header can hide
        inside a hunk.

        Args: None.
        Returns: None.
        """
        header = "diff --git a/x b/x\n"
        cases = (("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", "expected a file header"),
                 ("diff --git a/x b/y\n", "expected a file header"),
                 (header + "--- a/x\n", "without its"),
                 (header + "--- a/x\n+++ b/other\n", "disagree"),
                 (header + "--- a/x\n+++ b/x\n@@ junk @@\n", "damaged hunk"),
                 (header + "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n a\n", "ends early"),
                 (header + "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n-b\n+c\n", "longer"),
                 (header + "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
                  "--- a/.github/x\n+++ b/.github/x\n", "expected a file header"))
        for patch, message in cases:
            with self.subTest(message=message):
                self.refused(patch, message)

    def test_unsafe_and_protected_paths_are_refused(self):
        """Paths outside the project and under .github/ or .git/ are refused.

        Args: None.
        Returns: None.
        """
        self.refused("", "changes no files")
        for name in (".github/workflows/x.yml", ".git/config", "../x", "a/../../x",
                     "/etc/x", '"quoted"', "a\\b"):
            with self.subTest(name=name):
                self.refused(f"diff --git a/{name} b/{name}\nnew file mode 100644\n",
                             "may not change")

    def test_parse_issue_needs_both_markers(self):
        """An issue without the base and patch markers holds no change.

        Args: None.
        Returns: None.
        """
        for body in (None, "", "<!-- espi-change base=" + COMMIT + " -->"):
            with self.assertRaisesRegex(changes.ChangeError, "holds no change"):
                changes.parse_issue(body)


class SendTests(unittest.TestCase):
    """prepare_request, send and send_from_cli against a temporary copy."""

    def setUp(self):
        """Create a downloaded copy in a pretend prebuilt program.

        Args: None.
        Returns: None.
        """
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.folder = make_copy(self.root)
        patcher = prebuilt(self.root, feedback_token="tok")
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)

    def edit(self):
        """Make a text change and describe it.

        Args: None.
        Returns: None.
        """
        write(self.folder, "README.md", "# ESPI live\n\nA faster viewer.\n")
        write(self.folder, "docs/logo.png", b"\0new image")
        write(self.folder, "CHANGE_REQUEST.md", "Faster viewer\n\nIt says so now.")

    def test_only_a_downloaded_copy_can_send(self):
        """A folder without download bookkeeping is refused.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(changes.ChangeError, "not a source copy"):
            changes.prepare_request(self.root)

    def test_nothing_to_send(self):
        """No change, or only binary changes, cannot be sent.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(changes.ChangeError, r"differs from the original$"):
            changes.prepare_request(self.folder)
        write(self.folder, "docs/logo.png", b"\0changed")
        with self.assertRaisesRegex(changes.ChangeError, "binary files cannot be sent: "
                                                         "docs/logo.png"):
            changes.prepare_request(self.folder)

    def test_protected_changes_are_refused_before_sending(self):
        """A change to .github/ is refused on the user's machine already.

        Args: None.
        Returns: None.
        """
        write(self.folder, ".github/workflows/x.yml", "on: push\n")
        with self.assertRaisesRegex(changes.ChangeError, "may not change .github/"):
            changes.prepare_request(self.folder)

    def test_send_files_the_issue_records_it_and_tidies_up(self):
        """The issue holds the change; the send is recorded, the request file
        removed, and sending again reports the earlier issue.

        Args: None.
        Returns: None.
        """
        self.edit()
        with mock.patch.object(feedback, "create_issue",
                               return_value="https://x/issues/5") as create:
            self.assertEqual(changes.send(), ("https://x/issues/5", False))
            self.assertEqual(changes.send(), ("https://x/issues/5", True))
        create.assert_called_once()
        payload = create.call_args.args[0]
        self.assertEqual(payload["title"], "Faster viewer")
        base, patch, description = changes.parse_issue(payload["body"])
        self.assertEqual((base, description), (COMMIT, "It says so now."))
        self.assertEqual(changes.patch_paths(patch), ["README.md"])
        self.assertIn("`docs/logo.png`", payload["body"])
        self.assertFalse(os.path.exists(os.path.join(self.folder, "CHANGE_REQUEST.md")))

    def test_a_later_send_mentions_the_earlier_one(self):
        """Further changes go out as a new issue that names the earlier send.

        Args: None.
        Returns: None.
        """
        self.edit()
        with mock.patch.object(feedback, "create_issue", side_effect=[
                "https://x/issues/5", "https://x/issues/6"]) as create:
            changes.send(self.folder)
            write(self.folder, "README.md", "# ESPI live\n\nAn even faster viewer.\n")
            write(self.folder, "CHANGE_REQUEST.md", "Even faster")
            self.assertEqual(changes.send(self.folder), ("https://x/issues/6", False))
        self.assertIn("sent earlier in https://x/issues/5", create.call_args.args[0]["body"])

    def test_send_reports_github_failures_and_survives_bookkeeping_errors(self):
        """A refused issue raises ChangeError; a failed record does not.

        Args: None.
        Returns: None.
        """
        self.edit()
        with mock.patch.object(feedback, "create_issue",
                               side_effect=feedback.FeedbackError("no token")), \
                self.assertRaisesRegex(changes.ChangeError, "no token"):
            changes.send()
        with mock.patch.object(feedback, "create_issue", return_value="https://x/7"), \
                mock.patch.object(source_code, "record_sent", side_effect=OSError("ro")):
            self.assertEqual(changes.send(), ("https://x/7", False))

    def run_cli(self):
        """Run send_from_cli and capture its output.

        Args: None.
        Returns: (exit code, stdout, stderr).
        """
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = changes.send_from_cli()
        return code, out.getvalue(), err.getvalue()

    def test_cli_reports_each_outcome(self):
        """Failure, a new send and a repeat each print a clear line.

        Args: None.
        Returns: None.
        """
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Could not send the changes: nothing to send", err)
        self.edit()
        with mock.patch.object(feedback, "create_issue", return_value="https://x/9"):
            self.assertEqual(self.run_cli()[:2],
                             (0, "Sent. The ESPI live developers will review the change "
                                 "here: https://x/9\n"))
            self.assertEqual(self.run_cli()[:2],
                             (0, "These changes were already sent: https://x/9\n"))


class WorkflowCommandTests(unittest.TestCase):
    """main(), the command the codex-change workflow runs."""

    def setUp(self):
        """Create a temporary output folder.

        Args: None.
        Returns: None.
        """
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.args = ["extract", "--patch", os.path.join(self.folder, "p"),
                     "--description", os.path.join(self.folder, "d")]

    def run_main(self, body, runner=None):
        """Run the extract command with an issue body in the environment.

        Args: body is the issue text; runner replaces changes.main when set.
        Returns: (exit code, stdout, stderr).
        """
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"ISSUE_BODY": body}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = (runner or changes.main)(self.args)
        return code, out.getvalue(), err.getvalue()

    def test_extract_writes_patch_and_description(self):
        """A good issue yields the patch, the description and the base.

        Args: None.
        Returns: None.
        """
        code, out, _ = self.run_main(changes.issue_body(request()))
        self.assertEqual((code, out), (0, COMMIT + "\n"))
        with open(self.args[2], encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), PATCH)
        with open(self.args[4], encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Draw frames sooner.\n")

    def test_extract_explains_a_bad_issue(self):
        """An issue without a change exits 1 with the reason.

        Args: None.
        Returns: None.
        """
        code, _, err = self.run_main("just text")
        self.assertEqual(code, 1)
        self.assertIn("holds no change", err)

    def test_module_runs_as_a_script(self):
        """python -m espi_live.changes exits with main()'s status.

        Args: None.
        Returns: None.
        """
        def as_script(argv):
            """Run the module as __main__ with argv.

            Args: argv is the argument list.
            Returns: The SystemExit code.
            """
            with mock.patch.object(sys, "argv", ["changes"] + argv), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaises(SystemExit) as stop:
                    runpy.run_module("espi_live.changes", run_name="__main__")
            return stop.exception.code

        code, _, err = self.run_main("nothing", runner=as_script)
        self.assertEqual(code, 1)
        self.assertIn("holds no change", err)


if __name__ == "__main__":
    unittest.main()
