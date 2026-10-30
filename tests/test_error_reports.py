"""Camera-free checks for error reporting: fingerprints, scrubbing, delivery.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.parse
from unittest import mock

from espi_live import error_reports, paths
from list_log import ListLog

FIXED_FRAMES = [("src/espi_live/engine.py", "outer"), ("src/espi_live/engine.py", "inner")]
FINGERPRINT = "ESPI-ERR-v1-" + "a" * 12


class FakeResponse(io.BytesIO):
    """A urlopen result usable as a context manager."""

    def __enter__(self):
        """Enter the context.

        Args: None.
        Returns: This response.
        """
        return self

    def __exit__(self, *exc):
        """Leave the context.

        Args: exc is the exception triple.
        Returns: False, so exceptions propagate.
        """
        return False


class PySpinError(Exception):
    """A stand-in SpinnakerException that carries an error code."""

    __module__ = "PySpin"

    def __init__(self, code=None, message="failure"):
        """Create the exception.

        Args: code is the SDK error code (None to have none); message the text.
        Returns: None.
        """
        super().__init__(message)
        if code is not None:
            self.errorcode = code


def report_line(number, title, state="open", state_reason=None, closed_at=None):
    """Build one fake GitHub issue.

    Args: number is the issue number; title the title; state and state_reason
        and closed_at describe its state.
    Returns: An issue dict.
    """
    return {"number": number, "id": number * 10, "title": title, "state": state,
            "state_reason": state_reason, "closed_at": closed_at,
            "html_url": f"https://github.com/owner/repo/issues/{number}"}


class FakeGitHub:
    """A GitHub stand-in that records every change and answers from a list."""

    def __init__(self, issues=(), token="secret"):
        """Create the client.

        Args: issues are the issue dicts it knows; token is its token.
        Returns: None.
        """
        self.token = token
        self.issues = list(issues)
        self.created = []
        self.comments = []
        self.edits = []

    def reports(self):
        """List the known issues.

        Args: None.
        Returns: The issue dicts.
        """
        return list(self.issues)

    def matching(self, fingerprint):
        """Find the issues filed for a fingerprint.

        Args: fingerprint is the error ID.
        Returns: Matching issues, lowest number first.
        """
        prefix = f"[{fingerprint}]"
        return sorted((issue for issue in self.issues
                       if issue.get("title", "").startswith(prefix)),
                      key=lambda issue: issue["number"])

    def create(self, report):
        """Open an issue.

        Args: report is the ErrorReport.
        Returns: The new issue dict.
        """
        issue = report_line(77, report.title)
        issue["id"] = 1234
        self.created.append(report)
        self.issues.append(issue)
        return issue

    def comment(self, number, body):
        """Comment on an issue.

        Args: number is the issue number; body the text.
        Returns: The comment dict.
        """
        self.comments.append((number, body))
        return {}

    def edit(self, number, **fields):
        """Change an issue's state.

        Args: number is the issue number; fields are the PATCH body.
        Returns: The issue dict.
        """
        self.edits.append((number, fields))
        return {}


def make_report(**fields):
    """Build a complete ErrorReport for delivery tests.

    Args: fields override the defaults.
    Returns: An ErrorReport.
    """
    defaults = dict(
        fingerprint=FINGERPRINT, canonical="v1|ValueError||espi_live.engine:_loop",
        title=f"[{FINGERPRINT}] ValueError in espi_live.engine:_loop",
        error="ValueError: boom", where="engine", fatal=False, area="",
        body="body", seen="seen text", build=7, commit="abc1234def",
        committed_at="2026-02-01T00:00:00Z", created=time.time())
    defaults.update(fields)
    return error_reports.ErrorReport(**defaults)


def fingerprint_from_code(template, filename, entry="outer"):
    """Fingerprint an exception raised by code compiled with a given path.

    Args: template is module source defining entry() calling inner() which
        raises; filename is the path compiled into the code objects.
    Returns: The fingerprint of the raised exception.
    """
    namespace = {}
    exec(compile(template, filename, "exec"), namespace)
    try:
        namespace[entry]()
    except Exception as exc:
        return error_reports.fingerprint(exc)
    raise AssertionError("the code did not raise")


CODE_TEMPLATE = '''
def outer():
    inner()

def inner():
    raise ValueError("boom")
'''


class FingerprintTests(unittest.TestCase):
    """The deterministic error ID."""

    def test_path_parts_splits_both_separators(self):
        """Windows and POSIX separators both split; empty parts vanish.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.path_parts("C:\\a\\b\\c.py"), ["C:", "a", "b", "c.py"])
        self.assertEqual(error_reports.path_parts("/a//b/c.py"), ["a", "b", "c.py"])
        self.assertEqual(error_reports.path_parts(""), [])
        self.assertEqual(error_reports.path_parts(None), [])

    def test_in_app_module_maps_every_install_path(self):
        """Frozen, temporary and source layouts all name the same module.

        Args: None.
        Returns: None.
        """
        for filename in ("C:\\Users\\lab\\ESPI\\_internal\\espi_live\\engine.py",
                         "/tmp/_MEI1234/espi_live/engine.py",
                         "src/espi_live/engine.py"):
            self.assertEqual(error_reports.in_app_module(filename), "espi_live.engine")
        self.assertEqual(
            error_reports.in_app_module("/x/espi_live/camera/session.py"),
            "espi_live.camera.session")
        self.assertEqual(error_reports.in_app_module("/x/espi_live/__init__.py"),
                         "espi_live")
        self.assertIsNone(error_reports.in_app_module("/x/site-packages/numpy/core.py"))
        self.assertIsNone(error_reports.in_app_module("/x/espi_live"))
        self.assertIsNone(error_reports.in_app_module("/x/espi_live/readme"))

    def test_collapse_merges_recursion(self):
        """Consecutive repeats become one entry.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.collapse(["a", "a", "b", "a", "a"]), ["a", "b", "a"])
        self.assertEqual(error_reports.collapse([]), [])

    def test_frame_signature_keeps_innermost_in_app_frames(self):
        """More in-app frames than kept leaves the innermost ones.

        Args: None.
        Returns: None.
        """
        frames = [("src/espi_live/engine.py", f"f{index}") for index in range(12)]
        signature = error_reports.frame_signature(frames)
        self.assertEqual(signature, ">".join(f"espi_live.engine:f{index}"
                                             for index in range(4, 12)))

    def test_frame_signature_falls_back_to_file_names(self):
        """Without in-app frames the innermost three file names are used.

        Args: None.
        Returns: None.
        """
        frames = [("/usr/lib/numpy/core.py", "load"), ("qt.py", "paint"), ("", "hook")]
        self.assertEqual(error_reports.frame_signature(frames),
                         "core.py:load>qt.py:paint>?:hook")

    def test_frame_signature_skips_inlined_comprehensions(self):
        """Comprehension frames are filtered for 3.10/3.12 agreement.

        Args: None.
        Returns: None.
        """
        frames = [("src/espi_live/engine.py", "f"), ("<listcomp>", "<listcomp>"),
                  ("src/espi_live/engine.py", "g")]
        self.assertEqual(error_reports.frame_signature(frames),
                         "espi_live.engine:f>espi_live.engine:g")
        self.assertEqual(error_reports.frame_signature([]), "")

    def test_type_name_names_every_origin_the_same_way(self):
        """Builtins are bare, PySpin loses submodules, others keep the module.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.type_name(ValueError), "ValueError")
        self.assertEqual(error_reports.type_name(PySpinError), "PySpin.PySpinError")
        nested = type("Exc", (Exception,), {"__module__": "PySpin.SpinImage"})
        self.assertEqual(error_reports.type_name(nested), "PySpin.Exc")
        own = type("Own", (Exception,), {"__module__": "test_error_reports"})
        self.assertEqual(error_reports.type_name(own), "test_error_reports.Own")

    def test_stable_code_comes_only_from_pyspin(self):
        """The error code is kept for PySpin exceptions and dropped otherwise.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.stable_code(PySpinError(0x12)), "18")
        self.assertEqual(error_reports.stable_code(PySpinError()), "")
        self.assertEqual(error_reports.stable_code(ValueError("x")), "")

    def test_canonical_string_and_fingerprint(self):
        """The canonical text carries the code only when there is one.

        Args: None.
        Returns: None.
        """
        plain = error_reports.canonical_string("ValueError", "KeyError", "a>b")
        self.assertEqual(plain, "v1|ValueError|KeyError|a>b")
        coded = error_reports.canonical_string("ValueError", "", "a", "7")
        self.assertEqual(coded, "v1|ValueError||a|7")
        fingerprint = error_reports.fingerprint_of(plain)
        self.assertTrue(fingerprint.startswith("ESPI-ERR-v1-"))
        self.assertEqual(len(fingerprint), len("ESPI-ERR-v1-") + 12)
        self.assertEqual(fingerprint, error_reports.fingerprint_of(plain))

    def test_exception_frames_run_outermost_to_innermost(self):
        """The frame list starts at the caller and ends at the raise site.

        Args: None.
        Returns: None.
        """
        namespace = {}
        exec(compile(CODE_TEMPLATE, "src/espi_live/engine.py", "exec"), namespace)
        try:
            namespace["outer"]()
        except ValueError as exc:
            names = [function for _, function in error_reports.exception_frames(exc)]
        self.assertEqual(names[-2:], ["outer", "inner"])

    def test_exception_canonical_includes_the_cause(self):
        """Raising from another exception changes the canonical text.

        Args: None.
        Returns: None.
        """
        try:
            try:
                raise KeyError("k")
            except KeyError as cause:
                raise ValueError("boom") from cause
        except ValueError as exc:
            plain = error_reports.exception_canonical(ValueError("boom"))
            caused = error_reports.exception_canonical(exc)
        self.assertIn("|KeyError|", caused)
        self.assertNotIn("|KeyError|", plain)
        self.assertNotEqual(caused, plain)

    def test_install_path_and_line_shifts_keep_the_fingerprint(self):
        """The same code at three install paths and shifted lines agrees.

        Args: None.
        Returns: None.
        """
        base = fingerprint_from_code(CODE_TEMPLATE, "src/espi_live/engine.py")
        frozen = fingerprint_from_code(
            CODE_TEMPLATE, "C:\\Users\\lab\\ESPI\\_internal\\espi_live\\engine.py")
        temporary = fingerprint_from_code(CODE_TEMPLATE, "/tmp/_MEI99/espi_live/engine.py")
        shifted = fingerprint_from_code("\n\n" + CODE_TEMPLATE, "src/espi_live/engine.py")
        self.assertEqual(base, frozen)
        self.assertEqual(base, temporary)
        self.assertEqual(base, shifted)
        self.assertEqual(base, "ESPI-ERR-v1-5760d2b75c50")

    def test_every_difference_changes_the_fingerprint(self):
        """Another function, type, cause or code files under a new ID.

        Args: None.
        Returns: None.
        """
        base = fingerprint_from_code(CODE_TEMPLATE, "src/espi_live/engine.py")
        renamed = fingerprint_from_code(CODE_TEMPLATE.replace("outer", "alpha"),
                                        "src/espi_live/engine.py", entry="alpha")
        other_type = fingerprint_from_code(
            CODE_TEMPLATE.replace("ValueError", "RuntimeError"), "src/espi_live/engine.py")
        self.assertNotEqual(base, renamed)
        self.assertNotEqual(base, other_type)

        template = CODE_TEMPLATE.replace(
            'raise ValueError("boom")', 'raise ValueError("boom") from KeyError("k")')
        caused = fingerprint_from_code(template, "src/espi_live/engine.py")
        self.assertNotEqual(base, caused)
        self.assertNotEqual(error_reports.fingerprint(PySpinError(1, "m")),
                            error_reports.fingerprint(PySpinError(2, "m")))

    def test_message_signature_normalizes_volatile_text(self):
        """Numbers, paths, addresses and quotes become placeholders.

        Args: None.
        Returns: None.
        """
        signature = error_reports.message_signature(
            'Failed shader "mine.frag" at C:\\tmp\\a b.exe: 0xDEADBEEF, try 7')
        self.assertEqual(signature, 'Failed shader "" at <path> b.exe: #x, try #')
        self.assertEqual(len(error_reports.message_signature("x " * 200)), 120)


class CrashParsingTests(unittest.TestCase):
    """Reading faulthandler dumps from both platforms."""

    POSIX_DUMP = """
Fatal Python error: Segmentation fault

Current thread 0x00007f8b (most recent call first):
  File "/x/espi_live/engine.py", line 12 in _loop
  File "/x/espi_live/qt_app.py", line 34 in run
"""

    def test_posix_dump_gives_signal_and_frames(self):
        """A Linux/macOS dump yields the signal and the crashing thread.

        Args: None.
        Returns: None.
        """
        kind, frames = error_reports.parse_crash(self.POSIX_DUMP)
        self.assertEqual(kind, "native:SIGSEGV")
        self.assertEqual(frames, [("/x/espi_live/qt_app.py", "run"),
                                  ("/x/espi_live/engine.py", "_loop")])
        self.assertEqual(error_reports.frame_signature(frames),
                         "espi_live.qt_app:run>espi_live.engine:_loop")

    def test_windows_dump_and_other_thread_blocks(self):
        """A Windows dump names the exception; a Thread block parses too.

        Args: None.
        Returns: None.
        """
        windows = ("Windows fatal exception: access violation\n\n"
                   "Thread 0x1 (most recent call first):\n"
                   '  File "C:\\x\\espi_live\\engine.py", line 9 in _loop\n')
        kind, frames = error_reports.parse_crash(windows)
        self.assertEqual(kind, "native:ACCESS_VIOLATION")
        self.assertEqual(frames, [("C:\\x\\espi_live\\engine.py", "_loop")])
        kind, _ = error_reports.parse_crash("Fatal Python error: weird bus thing\n")
        self.assertEqual(kind, "native:WEIRD_BUS_THING")
        self.assertIsNone(error_reports.parse_crash("just a log line"))

    def test_unknown_headers(self):
        """Headerless text is not a crash; empty thread blocks are handled.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(error_reports.parse_crash("hello"))
        kind, frames = error_reports.parse_crash("Fatal Python error: Aborted\n")
        self.assertEqual(kind, "native:SIGABRT")
        self.assertEqual(frames, [])


class ScrubbingTests(unittest.TestCase):
    """Nothing personal reaches a public issue."""

    def test_personal_names_come_from_the_system(self):
        """User and host names are collected, filtered and deduplicated.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {"USER": "alexworland", "LOGNAME": "ab",
                                          "USERNAME": ""}), \
                mock.patch.object(error_reports.os.path, "expanduser",
                                  return_value="/home/espi"), \
                mock.patch.object(error_reports.getpass, "getuser", side_effect=KeyError), \
                mock.patch.object(error_reports.socket, "gethostname",
                                  return_value="cambox.lab.local"), \
                mock.patch.object(error_reports.platform, "node", return_value="cambox"):
            names = error_reports.personal_names()
        self.assertEqual(names, {"alexworland": "<user>", "cambox.lab.local": "<host>",
                                 "cambox": "<host>"})

    def test_folder_replacements_are_longest_first(self):
        """Program folders, the bundle and home are listed longest first.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = os.path.join(folder, "ESPI live")
            output = os.path.join(root, "output")
            with mock.patch.object(paths, "PROJECT_ROOT", root), \
                    mock.patch.object(paths, "OUTPUT_DIR", output), \
                    mock.patch.object(sys, "_MEIPASS",
                                      os.path.join(folder, "_MEI123"), create=True):
                folders = error_reports.folder_replacements()
        labels = [label for _, label in folders]
        self.assertEqual(sorted(labels), ["<bundle>", "<output>", "<program>", "~"])
        sizes = [len(path) for path, _ in folders]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_scrub_removes_every_kind_of_secret(self):
        """Folders, names, serials, addresses and tokens are replaced.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(error_reports, "folder_replacements", return_value=[
                ("/home/lab/proj/output", "<output>"), ("/home/lab/proj", "<program>")]), \
                mock.patch.object(error_reports, "personal_names",
                                  return_value={"lab": "<user>", "cambox": "<host>"}):
            scrub = error_reports.make_scrubber(("SN-12345678",))
        text = scrub("\n".join([
            "opened /home/lab/proj/output/frames/frame_001.png",
            'wrote C:\\home\\lab\\proj\\output\\recordings\\run.avi',
            "in C:\\Users\\lab\\AppData\\Local\\Temp\\_MEI1234\\espi_live",
            "token ghp_ABCDEFGHIJKLMNOPQRSTUVWX and github_pat_ABCDEFGHIJKLMNOPQRSTUVWX",
            "header Bearer abcdefghijklmnopqrst",
            "mail lab@example.com from cambox for user Lab",
            "camera Serial: SN-12345678 (S/N: ABC-1234)"]))
        for secret in ("/home/lab", "C:\\home\\lab", "_MEI1234", "ghp_ABCD",
                       "github_pat_ABCD", "abcdefghijklmnopqrst", "lab@example.com",
                       "cambox", "Lab", "SN-12345678", "ABC-1234"):
            self.assertNotIn(secret, text)
        self.assertIn("<output>/frames/<file>", text)
        self.assertIn("C:<output>\\recordings\\<file>", text)
        self.assertIn("<bundle>", text)
        self.assertIn("Serial: <serial>", text)
        self.assertIn("S/N: <serial>", text)
        self.assertIn("mail <email> from <host> for user <user>", text)
        self.assertIn("token <token> and <token>", text)
        self.assertIn("header Bearer <token>", text)

    def test_scrub_convenience_and_neutral_text(self):
        """The one-shot scrub() leaves text without anything personal alone.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.scrub("an unexpected error occurred"),
                         "an unexpected error occurred")

    def test_display_paths(self):
        """Package, site-packages and bare files each read clearly.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.display_path("C:\\x\\espi_live\\engine.py"),
                         "espi_live/engine.py")
        self.assertEqual(error_reports.display_path("/x/espi_live/__init__.py"),
                         "espi_live/__init__.py")
        self.assertEqual(error_reports.display_path("/x/espi_live/camera/session.py"),
                         "espi_live/camera/session.py")
        self.assertEqual(
            error_reports.display_path("/usr/lib/python3.12/site-packages/numpy/lib/fn.py"),
            "numpy/lib/fn.py")
        self.assertEqual(error_reports.display_path("engine.py"), "engine.py")
        self.assertEqual(error_reports.display_path(""), "")

    def test_relative_traceback_and_fence(self):
        """File paths shrink and triple backticks cannot break the block.

        Args: None.
        Returns: None.
        """
        text = error_reports.relative_traceback(
            'File "C:\\x\\espi_live\\engine.py", line 9 in _loop\n'
            'File "/usr/lib/python3.12/site-packages/numpy/lib/fn.py", line 2 in f\n')
        self.assertIn('File "espi_live/engine.py"', text)
        self.assertIn('File "numpy/lib/fn.py"', text)
        self.assertEqual(error_reports.fence("a ``` b"), "a ''' b")


class OutcomeTests(unittest.TestCase):
    """Labels and the one-line outcome text."""

    def test_labels_include_the_area_when_known(self):
        """A report with an area gains its label; others do not.

        Args: None.
        Returns: None.
        """
        self.assertEqual(make_report().labels, ["bug", "auto-error"])
        self.assertEqual(make_report(area="area: ui").labels,
                         ["bug", "auto-error", "area: ui"])

    def test_outcome_text_covers_every_kind(self):
        """Each outcome reads as one plain sentence.

        Args: None.
        Returns: None.
        """
        self.assertEqual(error_reports.outcome_text(None), "Reporting this error…")
        kinds = {
            "created": "Reported as #5",
            "commented": "Already reported as #5",
            "already": "Already reported as #5",
            "reopened": "Reported again as #5 (reopened)",
            "fixed": "Fixed in a newer build; update to get the fix",
            "declined": "Already known as #5",
            "browser": "Not sent automatically: report it in your browser",
            "saved": "Report saved; it will be sent next time"}
        for kind, text in kinds.items():
            outcome = error_reports.Outcome(FINGERPRINT, kind, 5, "https://x")
            self.assertEqual(error_reports.outcome_text(outcome), text)
        self.assertEqual(error_reports.outcome_text(error_reports.Outcome(FINGERPRINT, "odd")),
                         "odd")


class ExpectedTests(unittest.TestCase):
    """The allowlist of expected operating conditions."""

    def test_expected_types_match_through_the_mro(self):
        """A listed type or base class is expected; everything else is not.

        Args: None.
        Returns: None.
        """
        class DiskFull(OSError):
            """A stand-in OS error."""

        class UpdateError(Exception):
            """A stand-in updater error."""

        UpdateError.__module__ = "espi_live.updater"
        UpdateError.__qualname__ = "UpdateError"
        class FeedbackError(Exception):
            """A stand-in feedback error."""

        FeedbackError.__module__ = "espi_live.feedback"
        FeedbackError.__qualname__ = "FeedbackError"
        self.assertTrue(error_reports.is_expected(RuntimeError("x"), "camera_open"))
        self.assertTrue(error_reports.is_expected(DiskFull("x"), "save"))
        self.assertTrue(error_reports.is_expected(UpdateError("x"), "update"))
        self.assertTrue(error_reports.is_expected(FeedbackError("x"), "feedback"))
        self.assertFalse(error_reports.is_expected(ValueError("x"), "camera_open"))
        self.assertFalse(error_reports.is_expected(ValueError("x"), "engine"))


class SwitchTests(unittest.TestCase):
    """When reports are sent, and why not."""

    def test_parse_time_variants(self):
        """ISO times parse; anything else is None.

        Args: None.
        Returns: None.
        """
        parsed = error_reports.parse_time("2026-01-02T03:04:05Z")
        self.assertEqual(parsed.year, 2026)
        self.assertIsNotNone(parsed.tzinfo)
        naive = error_reports.parse_time("2026-01-02 03:04:05")
        self.assertIsNotNone(naive.tzinfo)
        self.assertIsNone(error_reports.parse_time(""))
        self.assertIsNone(error_reports.parse_time("soon"))
        self.assertIsNone(error_reports.parse_time(None))

    def test_resolve_enabled_matrix(self):
        """CLI, environment, the menu choice and builds decide together.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {error_reports.ENABLE_ENV: ""}):
            source = "off in source checkouts; set ESPI_ERROR_REPORTS=1 to send them"
            self.assertEqual(error_reports.resolve_enabled(), (False, source))
            self.assertEqual(error_reports.resolve_enabled(frozen=True), (True, None))
            self.assertEqual(error_reports.resolve_enabled(frozen=True, setting=False),
                             (False, None))
            self.assertEqual(error_reports.resolve_enabled(setting=False), (False, source))
        with mock.patch.dict(os.environ, {error_reports.ENABLE_ENV: "1"}):
            self.assertEqual(error_reports.resolve_enabled(), (True, None))
        with mock.patch.dict(os.environ, {error_reports.ENABLE_ENV: "0"}):
            self.assertEqual(error_reports.resolve_enabled(frozen=True),
                             (False, "turned off with ESPI_ERROR_REPORTS=0"))
        self.assertEqual(error_reports.resolve_enabled(cli_off=True, frozen=True),
                         (False, "turned off with --no-error-reports"))

    def test_write_json_is_atomic_and_cleans_up_on_failure(self):
        """Data lands atomically; a serialization failure removes the temp file.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "data.json")
            error_reports.write_json(path, {"a": 1})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"a": 1})
            with self.assertRaises(TypeError):
                error_reports.write_json(path, {"bad": {1, 2}})
            leftovers = [name for name in os.listdir(folder) if name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_write_json_survives_a_stuck_temp_file(self):
        """A temp file that cannot be removed does not mask the failure.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "data.json")
            with mock.patch.object(error_reports.os, "remove", side_effect=OSError("lock")):
                with self.assertRaises(TypeError):
                    error_reports.write_json(path, {"bad": {1, 2}})


class GitHubClientTests(unittest.TestCase):
    """The REST calls, against a patched urlopen."""

    def test_call_builds_the_request_and_parses_the_reply(self):
        """Method, path, query, token header and JSON reply line up.

        Args: None.
        Returns: None.
        """
        client = error_reports.GitHub("owner/repo", "secret")
        reply = json.dumps([{"number": 1}]).encode()
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(reply)) as urlopen:
            data = client.call("GET", "/issues", query={"labels": "auto-error", "page": 2})
        self.assertEqual(data, [{"number": 1}])
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url,
                         "https://api.github.com/repos/owner/repo/issues"
                         "?labels=auto-error&page=2")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

        anonymous = error_reports.GitHub("owner/repo", "")
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(reply)) as urlopen:
            anonymous.call("GET", "/issues")
        request = urlopen.call_args[0][0]
        self.assertIsNone(request.get_header("Authorization"))

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"{}")) as urlopen:
            client.call("POST", "/issues", {"title": "t"})
        request = urlopen.call_args[0][0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"title": "t"})
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_call_maps_errors_to_report_error(self):
        """HTTP, network and unreadable replies all raise ReportError.

        Args: None.
        Returns: None.
        """
        client = error_reports.GitHub("owner/repo", "")
        cases = ((urllib.error.HTTPError("u", 403, "no", {}, None),
                  "GitHub returned HTTP 403"),
                 (urllib.error.URLError("no route"), "cannot reach GitHub: no route"))
        for error, text in cases:
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(error_reports.ReportError, text):
                    client.call("GET", "/issues")
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"not json")):
            with self.assertRaisesRegex(error_reports.ReportError, "unreadable reply"):
                client.call("GET", "/issues")

    def test_reports_pages_and_skips_pull_requests(self):
        """Listing continues past full pages and drops pull requests.

        Args: None.
        Returns: None.
        """
        first = [report_line(number, f"t{number}") for number in range(100)]
        second = [report_line(200, "t200"), report_line(201, "t201")]
        first[-1]["pull_request"] = {"url": "x"}
        client = error_reports.GitHub("owner/repo", "")
        with mock.patch("urllib.request.urlopen",
                        side_effect=[FakeResponse(json.dumps(first).encode()),
                                     FakeResponse(json.dumps(second).encode())]) as urlopen:
            issues = client.reports()
        self.assertEqual(len(issues), 101)
        self.assertFalse(any("pull_request" in issue for issue in issues))
        self.assertEqual(urlopen.call_count, 2)

    def test_matching_filters_and_sorts(self):
        """Only the fingerprint's issues, lowest number first.

        Args: None.
        Returns: None.
        """
        client = error_reports.GitHub("owner/repo", "")
        client.issues = [report_line(9, f"[{FINGERPRINT}] second"),
                         report_line(4, f"[{FINGERPRINT}] first"),
                         report_line(5, "[ESPI-ERR-v1-" + "b" * 12 + "] other")]
        with mock.patch.object(client, "reports", return_value=list(client.issues)):
            self.assertEqual([issue["number"] for issue in client.matching(FINGERPRINT)],
                             [4, 9])

    def test_create_comment_and_edit_hit_their_endpoints(self):
        """The three write calls use POST/POST/PATCH with JSON bodies.

        Args: None.
        Returns: None.
        """
        client = error_reports.GitHub("owner/repo", "secret")
        report = make_report()
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(json.dumps(report_line(8, "t")).encode()
                                                  )) as urlopen:
            issue = client.create(report)
        self.assertEqual(issue["number"], 8)
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.github.com/repos/owner/repo/issues")
        self.assertEqual(json.loads(request.data)["labels"], report.labels)

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"{}")) as urlopen:
            client.comment(8, "seen again")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.github.com/repos/owner/repo/issues/8"
                                          "/comments")
        self.assertEqual(json.loads(request.data), {"body": "seen again"})

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"{}")) as urlopen:
            client.edit(8, state="open")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.get_method(), "PATCH")
        self.assertEqual(json.loads(request.data), {"state": "open"})


class ReporterBase(unittest.TestCase):
    """Shared setup: an enabled reporter over a temporary logs folder."""

    def setUp(self):
        """Create an enabled reporter in a temporary folder; pin the build.

        Args: None.
        Returns: None.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = temporary.name
        self.log_lines = []
        build_value = lambda name, default="": {  # noqa: E731
            "COMMIT": "abc1234def", "COMMITTED_AT": "2026-01-02T03:04:05Z",
            "REPO": "owner/repo"}.get(name, default)
        for target, value in (("build_value", build_value), ("current_build", lambda: 7),
                              ("describe_build", lambda: "ESPI live build 7")):
            patcher = mock.patch.object(error_reports.updater, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reporter = error_reports.ErrorReporter(
            enabled=True, folder=self.folder, log_tail=lambda: list(self.log_lines),
            uptime=lambda: 3.7)


class ReporterTests(ReporterBase):
    """Capturing errors into pending reports."""

    def test_update_context_and_scrubber(self):
        """Context fields are stored and serials reach the scrubber.

        Args: None.
        Returns: None.
        """
        self.reporter.update_context(front_end="Qt", serials=("SN-99",))
        self.assertEqual(self.reporter.context["front_end"], "Qt")
        self.assertEqual(self.reporter.scrubber()("camera serial SN-99"), "camera serial <serial>")

    def test_capture_saves_one_pending_report_per_error(self):
        """The first capture saves and announces; repeats return the same report.

        Args: None.
        Returns: None.
        """
        seen = []
        self.reporter.listeners.append(seen.append)
        try:
            raise ValueError("boom")
        except ValueError as exc:
            report = self.reporter.capture(exc, "engine", fatal=True)
            again = self.reporter.capture(exc, "engine")
            expected = error_reports.fingerprint(exc)
        self.assertIsNotNone(report)
        self.assertEqual(report.fingerprint, expected)
        self.assertTrue(report.fatal)
        self.assertEqual(report.where, "engine")
        path = os.path.join(self.reporter.pending_folder(), f"{report.fingerprint}.json")
        self.assertTrue(os.path.exists(path))
        self.assertIs(again, report)
        self.assertEqual([item.fingerprint for item in seen], [report.fingerprint])

    def test_capture_skips_disabled_and_expected_errors(self):
        """Nothing is saved when reporting is off or the error is expected.

        Args: None.
        Returns: None.
        """
        self.reporter.enabled = False
        try:
            raise RuntimeError("no camera")
        except RuntimeError as exc:
            self.assertIsNone(self.reporter.capture(exc, "engine"))
            self.reporter.enabled = True
            self.assertIsNone(self.reporter.capture(exc, "camera_open"))
        self.assertEqual(self.reporter.load_pending(), [])

    def test_capture_ignores_reentry(self):
        """A capture running inside a capture is skipped, not nested.

        Args: None.
        Returns: None.
        """
        nested = []

        def listener(report):
            """Try to capture a second error while the first is in progress.

            Args: report is the finished report.
            Returns: None.
            """
            try:
                raise RuntimeError("inner")
            except RuntimeError as exc:
                nested.append(self.reporter.capture(exc, "engine"))
        self.reporter.listeners.append(listener)
        try:
            raise ValueError("boom")
        except ValueError as exc:
            report = self.reporter.capture(exc, "engine")
        self.assertIsNotNone(report)
        self.assertEqual(nested, [None])
        self.assertEqual(len(self.reporter.load_pending()), 1)

    def test_capture_failures_are_logged_not_raised(self):
        """A bug inside the reporter itself is only logged.

        Args: None.
        Returns: None.
        """
        try:
            raise ValueError("boom")
        except ValueError as exc:
            with mock.patch.object(error_reports, "exception_canonical",
                                   side_effect=RuntimeError("no")):
                with self.assertLogs("espi_live.error_reports", "ERROR") as captured:
                    failed = self.reporter.capture(exc, "engine")
        self.assertIsNone(failed)
        self.assertIn("failed", captured.output[0])
        self.assertEqual(self.reporter.load_pending(), [])

    def test_capture_message_fingerprints_the_fixed_wording(self):
        """Two Qt messages differing only in numbers share one report.

        Args: None.
        Returns: None.
        """
        frames = [("C:\\x\\espi_live\\qt_app.py", "paintGL")]
        first = self.reporter.capture_message("critical", "shader failed, program 7",
                                              frames, "stack one")
        second = self.reporter.capture_message("critical", "shader failed, program 9",
                                               frames, "stack two")
        self.assertIs(first, second)
        self.assertIn("|qt:critical|", first.canonical)
        self.assertIn("Qt critical: shader failed", first.error)
        self.assertIn("stack one", first.body)
        fatal = self.reporter.capture_message("fatal", "abort for real", frames, "")
        self.assertTrue(fatal.fatal)
        self.assertIn("|qt:fatal|", fatal.canonical)

    def test_capture_crash_reports_a_dump(self):
        """A faulthandler dump becomes a fatal native-crash report.

        Args: None.
        Returns: None.
        """
        dump = ("Fatal Python error: Segmentation fault\n\n"
                "Current thread 0x1 (most recent call first):\n"
                '  File "/x/espi_live/engine.py", line 9 in _loop\n')
        report = self.reporter.capture_crash(dump)
        self.assertIsNotNone(report)
        self.assertTrue(report.fatal)
        self.assertEqual(report.where, "native_crash")
        self.assertIn("Native crash (SIGSEGV)", report.error)
        self.assertIn("|native:SIGSEGV|", report.canonical)
        self.assertIsNone(self.reporter.capture_crash("not a crash"))

    def test_build_assembles_the_report(self):
        """Title, area, scrubbed body and the seen-again note line up.

        Args: None.
        Returns: None.
        """
        canonical = "v1|ValueError|KeyError|espi_live.camera:grab>espi_live.engine:_loop"
        lines = ['Traceback (most recent call last):\n'
                 '  File "C:\\x\\espi_live\\engine.py", line 9 in _loop\n'
                 'ValueError: boom\n']
        self.reporter.update_context(front_end="Qt", display="GPU", source="camera BFS-U3",
                                     cameras=2, spinnaker="4.2.0.88")
        report = self.reporter.build(canonical, "ValueError: boom", lines, "engine", True,
                                     ["first log line"])
        self.assertTrue(report.title.startswith(f"[{report.fingerprint}] ValueError in "))
        self.assertTrue(report.title.endswith("espi_live.engine:_loop"))
        self.assertEqual(report.area, "")
        self.assertIn('File "espi_live/engine.py"', report.body)
        self.assertIn("**Error ID:**", report.body)
        self.assertIn(error_reports.FOOTER, report.body)
        self.assertIn("Spinnaker 4.2.0.88", report.body)
        self.assertIn("Front end: Qt, display GPU", report.body)
        self.assertIn("Running with: camera BFS-U3", report.body)
        self.assertIn("Cameras: 2", report.body)
        self.assertIn("Uptime: 3 s", report.body)
        self.assertIn("(fatal)", report.body)
        self.assertIn("### Last 1 log lines", report.body)
        self.assertIn("first log line", report.body)
        self.assertIn("build 7 (abc1234)", report.seen)
        self.assertIn("camera BFS-U3", report.seen)

    def test_build_area_comes_from_the_innermost_module(self):
        """The area label maps camera, ui and updates; others stay empty.

        Args: None.
        Returns: None.
        """
        cases = {"espi_live.camera_models:parse": "area: camera",
                 "espi_live.app:run": "area: ui",
                 "espi_live.qt_app:run": "area: ui",
                 "espi_live.updater:check": "area: updates",
                 "espi_live.engine:_loop": "",
                 "numpy.core:load": ""}
        for signature, area in cases.items():
            canonical = f"v1|ValueError||{signature}"
            self.assertEqual(self.reporter.build(
                canonical, "e", [], "engine", False, []).area, area, signature)

    def test_build_scrubs_the_error_and_log(self):
        """Home folders and saved file names never reach the body.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(error_reports, "folder_replacements", return_value=[
                ("/home/lab/proj/output", "<output>")]), \
                mock.patch.object(error_reports, "personal_names", return_value={}):
            report = self.reporter.build(
                "v1|OSError||espi_live.recording:write",
                "OSError: cannot write /home/lab/proj/output/frames/f.png",
                ["recorded /home/lab/proj/output/recordings/r.avi"], "recorder", False, [])
        self.assertNotIn("/home/lab", report.body)
        self.assertIn("<output>/frames/<file>", report.body)
        self.assertIn("<output>/recordings/<file>", report.body)

    def test_body_trims_the_log_first(self):
        """An oversized body loses log lines before any traceback.

        Args: None.
        Returns: None.
        """
        error = "ValueError: boom"
        facts = ["- Build: 7"]
        parts = ["Traceback (most recent call last):\nValueError: boom\n"]
        log_lines = [f"log {index} " + "y" * 120 for index in range(12)]
        with mock.patch.object(error_reports, "BODY_LIMIT", 600):
            body = error_reports.ErrorReporter.body(
                FINGERPRINT, "v1|ValueError||sig", error, facts, parts, log_lines)
        self.assertLessEqual(len(body), 600)
        self.assertNotIn("log 0", body)
        self.assertIn("log 11", body)

    def test_body_trims_outer_frames_then_single_pieces(self):
        """Outer File frames get a note; one long piece is cut in place.

        Args: None.
        Returns: None.
        """
        error = "ValueError: boom"
        facts = ["- Build: 7"]
        file_parts = [f'  File "a{index}.py", line {index} in f{index}\n' + "z" * 400
                      for index in range(2)]
        with mock.patch.object(error_reports, "BODY_LIMIT", 600):
            body = error_reports.ErrorReporter.body(
                FINGERPRINT, "v1|ValueError||sig", error, facts, list(file_parts), [])
            self.assertIn("outer frames trimmed", body)
            self.assertNotIn("a0.py", body)
            one_piece = error_reports.ErrorReporter.body(
                FINGERPRINT, "v1|ValueError||sig", error, facts, ["w" * 900], [])
        self.assertIn("... (trimmed)", one_piece)

    def test_pending_roundtrip_and_cleanup(self):
        """Pending files survive a reload; broken ones are dropped.

        Args: None.
        Returns: None.
        """
        report = make_report(created=2.0)
        self.reporter.save_pending(report)
        broken = os.path.join(self.reporter.pending_folder(), "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        notes = os.path.join(self.reporter.pending_folder(), "notes.txt")
        with open(notes, "w", encoding="utf-8") as handle:
            handle.write("ignore me")
        loaded = self.reporter.load_pending()
        self.assertEqual([item.fingerprint for _, item in loaded], [FINGERPRINT])
        self.assertEqual(loaded[0][1].error, "ValueError: boom")
        self.assertFalse(os.path.exists(broken))
        self.assertTrue(os.path.exists(notes))

    def test_prune_pending_keeps_the_newest_within_the_age(self):
        """More than 20 files or older than 30 days are removed.

        Args: None.
        Returns: None.
        """
        folder = self.reporter.pending_folder()
        now = time.time()
        for index in range(10):
            path = os.path.join(folder, f"ESPI-ERR-v1-{index:012d}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"x": index}, handle)
            age = 40 * 24 * 3600 if index in (1, 4, 7) else 60
            os.utime(path, (now - age, now - age))
        self.reporter.prune_pending(now=now)
        remaining = sorted(os.listdir(folder))
        self.assertEqual(remaining, [f"ESPI-ERR-v1-{index:012d}.json"
                                     for index in (0, 2, 3, 5, 6, 8, 9)])

    def test_prune_pending_tolerates_unreadable_entries(self):
        """A dangling entry and a stuck file do not stop the pruning.

        Args: None.
        Returns: None.
        """
        folder = self.reporter.pending_folder()
        for index in range(21):
            path = os.path.join(folder, f"ESPI-ERR-v1-{index:012d}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"n": index}, handle)
        os.symlink(os.path.join(folder, "missing"), os.path.join(folder, "dangling.json"))
        with mock.patch.object(error_reports.os, "remove", side_effect=OSError("locked")):
            self.reporter.prune_pending()
        self.assertEqual(len(os.listdir(folder)), 22)

    def test_load_pending_survives_an_undeletable_broken_file(self):
        """A broken report that cannot be removed is skipped, not fatal.

        Args: None.
        Returns: None.
        """
        broken = os.path.join(self.reporter.pending_folder(), "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("{")
        with mock.patch.object(error_reports.os, "remove", side_effect=OSError("locked")):
            self.assertEqual(self.reporter.load_pending(), [])
        self.assertTrue(os.path.exists(broken))

    def test_ledger_roundtrip_and_failures(self):
        """The ledger survives a reload; unreadable data resets; saves are quiet.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.reporter.load_ledger(), {"reports": {}})
        ledger_path = os.path.join(self.folder, error_reports.LEDGER_NAME)
        with open(ledger_path, "w", encoding="utf-8") as handle:
            handle.write("[")
        self.assertEqual(self.reporter.load_ledger(), {"reports": {}})
        with open(ledger_path, "w", encoding="utf-8") as handle:
            json.dump({"reports": {FINGERPRINT: {"issue": 4}}}, handle)
        self.assertEqual(self.reporter.load_ledger()["reports"][FINGERPRINT], {"issue": 4})
        with open(ledger_path, "w", encoding="utf-8") as handle:
            json.dump([1, 2], handle)
        self.assertEqual(self.reporter.load_ledger(), {"reports": {}})
        ledger = self.reporter.load_ledger()
        with mock.patch.object(error_reports, "write_json", side_effect=OSError("full")):
            with self.assertLogs("espi_live.error_reports", "WARNING"):
                self.reporter.save_ledger(ledger)

    def test_collect_native_crashes(self):
        """Earlier crash dumps are appended to their log, removed and reported.

        Args: None.
        Returns: None.
        """
        dump = ("Fatal Python error: Segmentation fault\n\n"
                "Current thread 0x1 (most recent call first):\n"
                '  File "/x/espi_live/engine.py", line 9 in _loop\n')
        crash = os.path.join(self.folder, "espi_1.crash")
        log = os.path.join(self.folder, "espi_1.log")
        with open(crash, "w", encoding="utf-8") as handle:
            handle.write(dump)
        with open(log, "w", encoding="utf-8") as handle:
            handle.write("Old camera (PRIVATE-SERIAL-1234) on old-private-host\n")
        empty = os.path.join(self.folder, "espi_2.crash")
        open(empty, "w", encoding="utf-8").close()
        orphan = os.path.join(self.folder, "espi_3.crash")
        with open(orphan, "w", encoding="utf-8") as handle:
            handle.write(dump.replace("_loop", "grab"))
        folder_crash = os.path.join(self.folder, "espi_4.crash")
        os.mkdir(folder_crash)
        with self.assertLogs("espi_live.error_reports", "WARNING"):
            reports = self.reporter.collect_native_crashes()
        self.assertEqual(len(reports), 2)
        for report in reports:
            self.assertNotIn("PRIVATE-SERIAL", report.body)
            self.assertNotIn("old-private-host", report.body)
        self.assertFalse(os.path.exists(crash))
        with open(log, encoding="utf-8") as handle:
            self.assertIn("Native crash (faulthandler)", handle.read())
        self.assertTrue(os.path.exists(empty))
        self.assertTrue(os.path.exists(orphan))
        self.assertEqual({report.canonical for report in reports},
                         {"v1|native:SIGSEGV||espi_live.engine:_loop",
                          "v1|native:SIGSEGV||espi_live.engine:grab"})

    def test_collect_native_crashes_skips_and_survives(self):
        """This run's own file is skipped; an unreadable folder yields nothing.

        Args: None.
        Returns: None.
        """
        dump = ("Fatal Python error: Segmentation fault\n\n"
                "Current thread 0x1 (most recent call first):\n"
                '  File "/x/espi_live/engine.py", line 9 in _loop\n')
        crash = os.path.join(self.folder, "espi_9.crash")
        with open(crash, "w", encoding="utf-8") as handle:
            handle.write(dump)
        self.assertEqual(self.reporter.collect_native_crashes(crash), [])
        blocker = os.path.join(self.folder, "blocked")
        open(blocker, "w", encoding="utf-8").close()
        other = error_reports.ErrorReporter(enabled=True, folder=blocker)
        self.assertEqual(other.collect_native_crashes(), [])


class DeliveryTests(ReporterBase):
    """The GitHub decision table, against a fake client."""

    def deliver(self, issues=(), token="secret", report=None, ledger=None):
        """Run deliver() against a FakeGitHub.

        Args: issues are the known issues; token the token; report and ledger
            overrides.
        Returns: (outcome, github, ledger).
        """
        github = FakeGitHub(issues=issues, token=token)
        ledger = {"reports": {}} if ledger is None else ledger
        outcome = self.reporter.deliver(report or make_report(), github, ledger)
        return outcome, github, ledger

    def test_no_match_creates_the_issue(self):
        """A new fingerprint opens an issue and records it in the ledger.

        Args: None.
        Returns: None.
        """
        outcome, github, ledger = self.deliver()
        self.assertEqual(outcome.kind, "created")
        self.assertEqual(outcome.number, 77)
        self.assertEqual(len(github.created), 1)
        entry = ledger["reports"][FINGERPRINT]
        self.assertEqual(entry, {"issue": 77, "url": outcome.url,
                                 "builds": [make_report().build]})

    def test_no_match_without_token_offers_the_browser(self):
        """Anonymous copies get a prefilled new-issue page instead.

        Args: None.
        Returns: None.
        """
        outcome, github, _ = self.deliver(token="")
        self.assertEqual(outcome.kind, "browser")
        self.assertTrue(outcome.url.startswith(
            "https://github.com/owner/repo/issues/new?title="))
        self.assertEqual(github.created, [])

    def test_open_issue_gets_one_comment_per_build(self):
        """An open issue gains a seen-again comment for a new build.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first")]
        outcome, github, ledger = self.deliver(issues=issues)
        self.assertEqual(outcome.kind, "commented")
        self.assertEqual(outcome.number, 4)
        self.assertEqual(github.comments, [(4, make_report().seen)])
        self.assertEqual(ledger["reports"][FINGERPRINT]["builds"], [7])

        outcome, again, ledger = self.deliver(issues=issues, ledger=ledger)
        self.assertEqual(outcome.kind, "already")
        self.assertEqual(again.comments, [])

    def test_open_issue_without_token_is_only_counted(self):
        """Anonymous copies never comment; they just record the issue.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first")]
        outcome, github, _ = self.deliver(issues=issues, token="")
        self.assertEqual(outcome.kind, "already")
        self.assertEqual(github.comments, [])

    def test_regression_is_reopened(self):
        """A closed-completed issue seen again in a newer build reopens.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first", state="closed",
                              state_reason="completed", closed_at="2026-01-01T00:00:00Z")]
        outcome, github, _ = self.deliver(issues=issues)
        self.assertEqual(outcome.kind, "reopened")
        self.assertEqual(github.edits, [(4, {"state": "open", "state_reason": "reopened"})])
        self.assertIn("Seen again in build 7", github.comments[0][1])

    def test_older_build_leaves_a_fixed_issue_alone(self):
        """A build older than the closure only hears that it is fixed.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first", state="closed",
                              state_reason="completed", closed_at="2026-06-01T00:00:00Z")]
        outcome, github, _ = self.deliver(issues=issues,
                                          report=make_report(committed_at="2026-01-01T00:00:00Z"))
        self.assertEqual(outcome.kind, "fixed")
        self.assertEqual(github.edits, [])
        self.assertEqual(github.comments, [])

    def test_regression_without_token_offers_the_browser(self):
        """An anonymous copy hears about a regression without reopening it.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first", state="closed",
                              state_reason="completed", closed_at="2026-01-01T00:00:00Z")]
        outcome, github, _ = self.deliver(issues=issues, token="")
        self.assertEqual(outcome.kind, "browser")
        self.assertEqual(outcome.number, 4)
        self.assertEqual(github.edits, [])
        self.assertEqual(github.comments, [])

    def test_not_planned_and_undated_closures_are_left_alone(self):
        """A wontfix issue and one with no closing time are not touched.

        Args: None.
        Returns: None.
        """
        issues = [report_line(4, f"[{FINGERPRINT}] first", state="closed",
                              state_reason="not_planned")]
        outcome, github, _ = self.deliver(issues=issues)
        self.assertEqual(outcome.kind, "declined")
        outcome, github, _ = self.deliver(issues=issues)
        self.assertEqual(outcome.kind, "declined")

        issues = [report_line(5, f"[{FINGERPRINT}] first", state="closed")]
        outcome, github, _ = self.deliver(issues=issues,
                                          report=make_report(committed_at="2026-01-01T00:00:00Z"))
        self.assertEqual(outcome.kind, "fixed")
        self.assertEqual(github.edits, [])

    def test_the_race_settles_on_the_lowest_number(self):
        """A duplicate created by this copy is closed in favour of the first.

        Args: None.
        Returns: None.
        """
        github = FakeGitHub(issues=[report_line(3, f"[{FINGERPRINT}] first")])
        created = dict(report_line(77, f"[{FINGERPRINT}] mine"), id=1234)
        kept = self.reporter._settle_race(github, make_report(), created)
        self.assertEqual(kept["number"], 3)
        self.assertEqual(github.comments, [(77, "Duplicate of #3")])
        self.assertEqual(github.edits, [(77, {"state": "closed", "state_reason": "duplicate",
                                              "duplicate_issue_id": 30})])

        github = FakeGitHub()
        self.assertEqual(self.reporter._settle_race(github, make_report(), created)["number"],
                         77)
        self.assertEqual(github.comments, [])

        failing = FakeGitHub(issues=[report_line(3, f"[{FINGERPRINT}] first")])
        failing.edit = mock.MagicMock(side_effect=error_reports.ReportError("read only"))
        with self.assertLogs("espi_live.error_reports", "WARNING"):
            kept = self.reporter._settle_race(failing, make_report(), created)
        self.assertEqual(kept["number"], 3)

    def test_send_pending_delivers_and_cleans_up(self):
        """Every pending report is delivered once, then its file is removed.

        Args: None.
        Returns: None.
        """
        first = make_report()
        second = make_report(fingerprint="ESPI-ERR-v1-" + "b" * 12,
                             canonical="v1|ValueError||other")
        self.reporter.save_pending(first)
        self.reporter.save_pending(second)
        github = FakeGitHub(issues=[report_line(4, f"[{second.fingerprint}] earlier")])
        posted = []
        with mock.patch.object(error_reports, "GitHub", lambda repo, token, **kwargs: github), \
                mock.patch.object(error_reports.feedback, "feedback_token",
                                  return_value="secret"), \
                mock.patch.object(error_reports.feedback, "repository",
                                  return_value="owner/repo"):
            self.reporter.send_pending(posted.append)
        self.assertEqual([outcome.kind for _, outcome in posted], ["created", "commented"])
        self.assertEqual(os.listdir(self.reporter.pending_folder()), [])
        ledger = self.reporter.load_ledger()
        self.assertEqual(ledger["reports"][first.fingerprint]["issue"], 77)
        self.assertEqual(ledger["reports"][second.fingerprint]["issue"], 4)
        self.assertIn(second.fingerprint, self.reporter.delivered)

    def test_send_pending_respects_the_limits(self):
        """At most five reports per run; already-sent ones are skipped.

        Args: None.
        Returns: None.
        """
        for index in range(7):
            self.reporter.save_pending(make_report(
                fingerprint=f"ESPI-ERR-v1-{index:012d}"))
        posted = []
        with mock.patch.object(error_reports, "MAX_DELIVERIES_PER_RUN", 2), \
                mock.patch.object(error_reports, "GitHub", lambda repo, token, **kwargs: FakeGitHub()), \
                mock.patch.object(error_reports.feedback, "feedback_token",
                                  return_value="secret"), \
                mock.patch.object(error_reports.feedback, "repository",
                                  return_value="owner/repo"):
            self.reporter.send_pending(posted.append)
            again = []
            self.reporter.send_pending(again.append)
            self.reporter.save_pending(make_report(fingerprint="ESPI-ERR-v1-000000000000",
                                                   created=1.0))
            third = []
            self.reporter.send_pending(third.append)
        self.assertEqual(len(posted), 2)
        self.assertEqual(again, [])
        self.assertEqual(third, [])
        self.assertTrue(os.path.exists(os.path.join(
            self.reporter.pending_folder(), "ESPI-ERR-v1-000000000000.json")))

    def test_send_pending_keeps_reports_that_could_not_be_sent(self):
        """A GitHub failure leaves the file pending and posts a saved outcome.

        Args: None.
        Returns: None.
        """
        self.reporter.save_pending(make_report())
        github = FakeGitHub()
        github.matching = mock.MagicMock(side_effect=error_reports.ReportError("offline"))
        posted = []
        with mock.patch.object(error_reports, "GitHub", lambda repo, token, **kwargs: github), \
                mock.patch.object(error_reports.feedback, "feedback_token",
                                  return_value="secret"), \
                mock.patch.object(error_reports.feedback, "repository",
                                  return_value="owner/repo"), \
                self.assertLogs("espi_live.error_reports", "WARNING"):
            self.reporter.send_pending(posted.append)
        self.assertEqual([outcome.kind for _, outcome in posted], ["saved"])
        self.assertEqual(len(os.listdir(self.reporter.pending_folder())), 1)

    def test_send_pending_survives_a_stuck_file(self):
        """A delivered report whose file cannot be removed still posts.

        Args: None.
        Returns: None.
        """
        self.reporter.save_pending(make_report())
        posted = []
        with mock.patch.object(error_reports, "GitHub", lambda repo, token, **kwargs: FakeGitHub()), \
                mock.patch.object(error_reports.feedback, "feedback_token",
                                  return_value="secret"), \
                mock.patch.object(error_reports.feedback, "repository",
                                  return_value="owner/repo"), \
                mock.patch.object(error_reports.os, "remove", side_effect=OSError("lock")):
            self.reporter.send_pending(posted.append)
        self.assertEqual([outcome.kind for _, outcome in posted], ["created"])
        self.assertEqual(len(os.listdir(self.reporter.pending_folder())), 1)

    def test_send_pending_does_nothing_when_off_or_empty(self):
        """A disabled reporter or an empty folder makes no calls at all.

        Args: None.
        Returns: None.
        """
        posted = []
        with mock.patch.object(error_reports, "GitHub") as github_class:
            self.reporter.enabled = False
            self.reporter.send_pending(posted.append)
            self.reporter.enabled = True
            self.reporter.send_pending(posted.append)
        github_class.assert_not_called()
        self.assertEqual(posted, [])

    def test_outcomes_are_logged_with_an_opt_out_hint_once(self):
        """The first sent report explains how to turn reporting off.

        Args: None.
        Returns: None.
        """
        report = make_report()
        outcome = error_reports.Outcome(FINGERPRINT, "created", 5,
                                        "https://github.com/owner/repo/issues/5")
        with self.assertLogs("espi_live.error_reports", "INFO") as captured:
            self.reporter._log_outcome(report, outcome)
            self.reporter._log_outcome(report, error_reports.Outcome(
                FINGERPRINT, "browser", url="https://github.com/owner/repo/issues/new"))
        text = "\n".join(captured.output)
        self.assertIn("Reported as #5", text)
        self.assertIn(error_reports.TURN_OFF_HINT, text)
        self.assertEqual(text.count("Report it in a browser"), 1)

    def test_browser_url_fits_the_limit(self):
        """A long body is shortened until the URL fits.

        Args: None.
        Returns: None.
        """
        report = make_report(body="long " * 3000)
        with mock.patch.object(error_reports.feedback, "repository",
                               return_value="owner/repo"), \
                mock.patch.object(error_reports, "BROWSER_URL_LIMIT", 400):
            url = error_reports.browser_url(report)
        self.assertLessEqual(len(url), 400)
        self.assertIn("trimmed", url)
        self.assertIn("labels=bug%2Cauto-error", url)


class StubReporter:
    """A stand-in reporter for the worker thread."""

    def __init__(self, outcome=None, failure=None):
        """Create the stub.

        Args: outcome is posted once per send_pending call; failure, when
            given, is raised instead.
        Returns: None.
        """
        self.outcome = outcome or error_reports.Outcome(FINGERPRINT, "created", 5)
        self.failure = failure
        self.calls = []
        self.done = threading.Event()

    def send_pending(self, post):
        """Post one outcome or fail, then signal the test.

        Args: post is the callback that receives events.
        Returns: None.
        """
        self.calls.append(post)
        if self.failure is not None:
            self.done.set()
            raise self.failure
        post(("outcome", self.outcome))
        self.done.set()


class WorkerTests(unittest.TestCase):
    """The background sender thread."""

    def test_worker_sends_pending_then_idles_until_stopped(self):
        """A kick delivers pending reports; stop() ends the thread.

        Args: None.
        Returns: None.
        """
        stub = StubReporter()
        worker = error_reports.ErrorReportWorker(stub)
        worker.start()
        self.assertTrue(stub.done.wait(5))
        kind, outcome = worker.events.get_nowait()
        self.assertEqual(kind, "outcome")
        self.assertEqual(outcome.number, 5)
        worker.stop()
        worker.thread.join(5)
        self.assertFalse(worker.thread.is_alive())

    def test_worker_survives_reporter_failures(self):
        """An exception in the sender is logged and the thread keeps going.

        Args: None.
        Returns: None.
        """
        stub = StubReporter(failure=RuntimeError("disk gone"))
        worker = error_reports.ErrorReportWorker(stub)
        with self.assertLogs("espi_live.error_reports", "ERROR"):
            worker.start()
            self.assertTrue(stub.done.wait(5))
            worker.stop()
        worker.thread.join(5)
        self.assertFalse(worker.thread.is_alive())

    def test_kick_and_stop_are_safe_before_a_start(self):
        """Kicks and stops without a thread change nothing.

        Args: None.
        Returns: None.
        """
        worker = error_reports.ErrorReportWorker(StubReporter())
        worker.kick()
        worker.stop()
        self.assertIsNone(worker.thread)


class HookTests(ReporterBase):
    """The process-wide error hooks."""

    def install(self):
        """Install the hooks and schedule their removal.

        Args: None.
        Returns: The Hooks object.
        """
        hooks = error_reports.install_hooks(self.reporter)
        self.addCleanup(hooks.uninstall)
        return hooks

    def test_install_and_uninstall_restore_everything(self):
        """The three hooks and the module reporter are saved and restored.

        Args: None.
        Returns: None.
        """
        before = (sys.excepthook, threading.excepthook, sys.unraisablehook)
        hooks = self.install()
        self.assertEqual(sys.excepthook, hooks.excepthook)
        self.assertEqual(threading.excepthook, hooks.thread_excepthook)
        self.assertEqual(sys.unraisablehook, hooks.unraisablehook)
        self.assertIs(error_reports.current_reporter(), self.reporter)
        hooks.uninstall()
        self.assertEqual((sys.excepthook, threading.excepthook, sys.unraisablehook), before)
        self.assertIsNone(error_reports.current_reporter())

    def test_excepthook_reports_uncaught_errors(self):
        """A main-thread exception is logged and captured.

        Args: None.
        Returns: None.
        """
        hooks = self.install()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            expected_id = error_reports.fingerprint(exc)
            with self.assertLogs("espi_live.error_reports", "ERROR"):
                hooks.excepthook(ValueError, exc, exc.__traceback__)
        self.assertEqual([item.fingerprint for _, item in self.reporter.load_pending()],
                         [expected_id])

    def test_excepthook_passes_interrupts_through_and_fills_tracebacks(self):
        """KeyboardInterrupt keeps its default handler; tb-less errors still work.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(sys, "excepthook") as default:
            hooks = self.install()
            try:
                hooks.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
                default.assert_called_once()
                hooks.excepthook(ValueError, ValueError("bare"), None)
            finally:
                hooks.uninstall()
        self.assertEqual(len(self.reporter.load_pending()), 1)

    def test_thread_hook_reports_worker_threads_but_not_its_own(self):
        """Other threads are reported; SystemExit and the sender are not.

        Args: None.
        Returns: None.
        """
        hooks = self.install()
        sender = threading.Thread(target=lambda: None, name=error_reports.WORKER_THREAD)
        worker = threading.Thread(target=lambda: None, name="engine")

        def run(exc):
            """Build hook arguments for a thread.

            Args: exc is the exception value.
            Returns: An ExceptHookArgs.
            """
            return threading.ExceptHookArgs((type(exc), exc, exc.__traceback__, None))

        with self.assertNoLogs("espi_live.error_reports"):
            hooks.thread_excepthook(threading.ExceptHookArgs(
                (SystemExit, SystemExit(), None, worker)))
        with self.assertLogs("espi_live.error_reports", "ERROR"):
            hooks.thread_excepthook(threading.ExceptHookArgs(
                (ValueError, ValueError("in sender"), None, sender)))
        self.assertEqual(self.reporter.load_pending(), [])
        with self.assertLogs("espi_live.error_reports", "ERROR"):
            hooks.thread_excepthook(run(ValueError("in worker")))
        self.assertEqual(len(self.reporter.load_pending()), 1)
        with self.assertLogs("espi_live.error_reports", "ERROR"):
            hooks.thread_excepthook(threading.ExceptHookArgs(
                (ValueError, None, None, worker)))
        self.assertEqual(len(self.reporter.load_pending()), 1)

    def test_unraisablehook_reports_finalizer_errors(self):
        """An ignored exception is logged and captured; a bare one is not.

        Args: None.
        Returns: None.
        """
        hooks = self.install()
        try:
            raise ValueError("late failure")
        except ValueError as exc:
            args = types.SimpleNamespace(err_msg=None, object=object(), exc_type=ValueError,
                                         exc_value=exc, exc_traceback=None)
            with self.assertLogs("espi_live.error_reports", "ERROR") as captured:
                hooks.unraisablehook(args)
            self.assertIn("Exception ignored", captured.output[0])
            self.assertEqual(len(self.reporter.load_pending()), 1)
            with self.assertLogs("espi_live.error_reports", "ERROR"):
                hooks.unraisablehook(types.SimpleNamespace(
                    err_msg="bad news", object=object(), exc_type=ValueError,
                    exc_value=None, exc_traceback=None))
        self.assertEqual(len(self.reporter.load_pending()), 1)

    def test_capture_and_caught_route_through_the_reporter(self):
        """capture() uses the installed reporter; caught() picks the level.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(error_reports.capture(ValueError("x"), "engine"))
        hooks = self.install()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            report = error_reports.capture(exc, "engine", fatal=True)
            self.assertIsNotNone(report)
            quiet = ListLog()
            try:
                raise RuntimeError("no camera")
            except RuntimeError as condition:
                expected = error_reports.caught(quiet, condition, "camera_open",
                                                "open failed")
            loud = ListLog()
            unexpected = error_reports.caught(loud, exc, "engine", "worker died")
            original = exc
        self.assertIsNone(expected)
        self.assertEqual(quiet.at("WARNING"), ["open failed: no camera"])
        self.assertIsNotNone(unexpected)
        self.assertEqual(loud.records[0][0], "ERROR")
        self.assertEqual(loud.records[0][2], original)
        self.assertEqual(error_reports.current_reporter(), self.reporter)
        self.assertEqual(hooks.reporter, self.reporter)


class ReviewRegressionTests(ReporterBase):
    """Privacy and delivery regressions found during PR review."""

    def test_all_public_fields_are_scrubbed(self):
        """Context, thread names, titles and comments use the scrubber.

        Args: None.
        Returns: None.
        """
        secret = 'private@example.com'
        self.reporter.update_context(source='camera ' + secret, display=secret,
                                     serials=['SN-123456'])
        with mock.patch.object(threading.current_thread(), '_name', secret):
            report = self.reporter.build(
                'v1|ValueError||private@example.com.py:inner', 'ValueError: boom',
                [], 'engine', False, ['Saved: /tmp/Frames/patient_case.png'])
        for text in (report.title, report.body, report.seen):
            self.assertNotIn(secret, text)
            self.assertNotIn('patient_case', text)
        github = FakeGitHub()
        report.title = report.body = report.seen = secret
        self.reporter.deliver(report, github, {'reports': {}})
        self.assertEqual(github.created[0].body, '<email>')
        github.created.clear()
        github.issues[0]['title'] = f'[{report.fingerprint}] existing'
        self.reporter.deliver(report, github, {'reports': {}})
        self.assertEqual(github.comments[0][1], '<email>')

    def test_serials_are_retained_after_context_changes(self):
        """Disconnected cameras remain scrubbed from the recent log tail.

        Args: None.
        Returns: None.
        """
        self.reporter.update_context(serials=['SN-123456'])
        self.reporter.update_context(serials=['SN-987654'])
        self.assertEqual(self.reporter.scrubber()('[SN-123456] SN-987654'),
                         '[<serial>] <serial>')

    def test_send_pending_stops_after_opt_out(self):
        """An opt-out during a request leaves the rest of the batch pending.

        Args: None.
        Returns: None.
        """
        first = make_report()
        second = make_report(fingerprint='ESPI-ERR-v1-bbbbbbbbbbbb')
        self.reporter.save_pending(first)
        self.reporter.save_pending(second)
        github = FakeGitHub()
        posted = []

        def opt_out(event):
            """Disable reporting after the first outcome.

            Args: event is the completed outcome.
            Returns: None.
            """
            posted.append(event)
            self.reporter.enabled = False

        with mock.patch.object(error_reports, 'GitHub', return_value=github), \
                mock.patch.object(error_reports.feedback, 'feedback_token', return_value='fake'):
            self.reporter.send_pending(opt_out)
        self.assertEqual(len(posted), 1)
        self.assertEqual([report.fingerprint for _, report in self.reporter.load_pending()],
                         [second.fingerprint])

    def test_consent_is_rechecked_after_lookup_before_mutation(self):
        """Disabling during lookup prevents the following issue POST.

        Args: None.
        Returns: None.
        """
        self.reporter.save_pending(make_report())
        methods = []

        def respond(request, **kwargs):
            """Return an empty lookup and revoke consent while it is in flight.

            Args: request is the HTTP request; kwargs are transport options.
            Returns: An empty issue-list response.
            """
            methods.append(request.method)
            self.reporter.enabled = False
            return FakeResponse(b'[]')

        with mock.patch.object(error_reports.urllib.request, 'urlopen', side_effect=respond), \
                mock.patch.object(error_reports.feedback, 'feedback_token', return_value='fake'), \
                self.assertLogs('espi_live.error_reports', 'WARNING'):
            self.reporter.send_pending(lambda event: None)
        self.assertEqual(methods, ['GET'])
        self.assertEqual(len(self.reporter.load_pending()), 1)

    def test_old_pending_reports_are_pruned_before_startup_delivery(self):
        """Expiry applies even when startup has captured no new errors.

        Args: None.
        Returns: None.
        """
        self.reporter.save_pending(make_report())
        path, _ = self.reporter.load_pending()[0]
        old = time.time() - error_reports.PENDING_MAX_AGE_S - 100
        os.utime(path, (old, old))
        with mock.patch.object(error_reports, 'GitHub') as client:
            self.reporter.send_pending(lambda event: None)
        client.assert_not_called()
        self.assertEqual(self.reporter.load_pending(), [])

    def test_live_native_crash_file_is_not_consumed(self):
        """A surviving native exception in another active run stays local.

        Args: None.
        Returns: None.
        """
        from espi_live.logs import RunLock
        path = os.path.join(self.folder, 'espi_active.crash')
        with open(path, 'w') as handle:
            handle.write('Fatal Python error: Segmentation fault\n')
        lock = RunLock(path[:-6] + '.log')
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.close)
        self.assertEqual(self.reporter.collect_native_crashes(), [])
        self.assertTrue(os.path.exists(path))

    def test_v1_golden_cause_code_and_comprehensions(self):
        """Literal digests pin the v1 serialization and comprehension filtering.

        Args: None.
        Returns: None.
        """
        cases = [
            ('ValueError', 'KeyError',
             [('src/espi_live/engine.py', 'outer'), ('src/espi_live/engine.py', 'inner')],
             '', 'v1|ValueError|KeyError|espi_live.engine:outer>espi_live.engine:inner',
             'ESPI-ERR-v1-e55ae320525a'),
            ('PySpin.PySpinError', '', [('src/espi_live/camera.py', 'grab')], '-1011',
             'v1|PySpin.PySpinError||espi_live.camera:grab|-1011', 'ESPI-ERR-v1-85d9976f0f50'),
            ('ValueError', '', [('src/espi_live/engine.py', 'f'),
                               ('src/espi_live/engine.py', '<listcomp>'),
                               ('src/espi_live/engine.py', 'g')], '',
             'v1|ValueError||espi_live.engine:f>espi_live.engine:g', 'ESPI-ERR-v1-a63b22e0e311')]
        for kind, cause, frames, code, expected, digest in cases:
            actual = error_reports.canonical_string(
                kind, cause, error_reports.frame_signature(frames), code)
            self.assertEqual(actual, expected)
            self.assertEqual(error_reports.fingerprint_of(actual), digest)
