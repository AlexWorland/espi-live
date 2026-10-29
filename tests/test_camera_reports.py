"""Network-free checks for camera reports filed as GitHub issues.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from espi_live import camera_reports, feedback, paths
from espi_live.camera import CameraInfo
from espi_live.capture_models import CaptureIdentity
from test_feedback import FakeResponse

WEBCAM_INFO = CameraInfo("video 1", "Webcam C270", kind="capture", index=700,
                         backend=700, vid=0x046D, pid=0x0825, path="usb#vid_046d")
WEBCAM = CaptureIdentity(name="Webcam C270", backend="DSHOW", path="usb#vid_046d",
                         vid=0x046D, pid=0x0825, width=640, height=480,
                         fourcc="MJPG", fps=30.0)
DETAILS = (("Device name", "Webcam C270"), ("FOURCC", "MJPG"))


def webcam_report(details=DETAILS, identity=WEBCAM, info=WEBCAM_INFO):
    """Build the report for the fake webcam.

    Args: details, identity and info are passed to capture_report.
    Returns: A Report.
    """
    return camera_reports.capture_report(info, identity, details)


def reply(data):
    """Wrap JSON data as a urlopen result.

    Args: data is the object GitHub would return.
    Returns: A FakeResponse.
    """
    return FakeResponse(json.dumps(data).encode())


class ReportTextTests(unittest.TestCase):
    """Keys, titles, Markdown and size limits."""

    def test_report_key_and_title(self):
        """The key is name plus USB ID, else the name, else the device id.

        Args: None.
        Returns: None.
        """
        report = webcam_report()
        self.assertEqual(report.key, "Webcam C270 (046d:0825)")
        self.assertEqual(report.title, "Camera report: Webcam C270 (046d:0825)")
        virtual = CameraInfo("video 2", "OBS Virtual Camera", kind="capture")
        self.assertEqual(camera_reports.capture_report(
            virtual, CaptureIdentity(name="OBS Virtual Camera"), ()).key,
            "OBS Virtual Camera")
        nameless = CameraInfo("video 3", "", kind="capture")
        self.assertEqual(camera_reports.capture_report(nameless, CaptureIdentity(), ()).key,
                         "video 3")

    def test_body_holds_description_details_context_and_marker(self):
        """The body describes the device, lists its details and ends with the marker.

        Args: None.
        Returns: None.
        """
        body = webcam_report().body
        self.assertIn("Webcam C270 (video capture)", body)
        self.assertIn("Video: 640 × 480, MJPG", body)
        self.assertIn("not suitable for ESPI", body)
        self.assertIn("| FOURCC | MJPG |", body)
        self.assertIn("- Build: ", body)
        self.assertIn("- Platform: ", body)
        self.assertTrue(body.endswith(camera_reports.marker("Webcam C270 (046d:0825)")))
        self.assertIn("USB 046d:0825", webcam_report().summary)

    def test_markdown_breaking_values_escaped(self):
        """Pipes, newlines and backticks cannot break the details table.

        Args: None.
        Returns: None.
        """
        body = webcam_report(details=(("Odd", "a|b\n```c"),)).body
        rows = [line for line in body.splitlines() if line.startswith("| Odd |")]
        self.assertEqual(len(rows), 1)
        self.assertIn("a\\|b", rows[0])
        self.assertNotIn("```", rows[0])
        self.assertEqual(camera_reports.cell("x\r\ny"), "x y")

    def test_oversized_details_trimmed(self):
        """A huge details table is cut to fit and says how much was left out.

        Args: None.
        Returns: None.
        """
        details = tuple((f"Property {n}", "x" * 100) for n in range(5000))
        body = webcam_report(details=details).body
        self.assertLessEqual(len(body), camera_reports.MAX_BODY)
        self.assertIn("more rows not shown", body)
        self.assertTrue(body.endswith(camera_reports.marker("Webcam C270 (046d:0825)")))

    def test_browser_url_short(self):
        """The prefilled page fits a URL even for an absurdly long name.

        Args: None.
        Returns: None.
        """
        url = camera_reports.browser_url(webcam_report())
        self.assertIn("labels=camera-support", url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(query["title"], ["Camera report: Webcam C270 (046d:0825)"])
        long_name = "N" * 10000
        huge = webcam_report(identity=CaptureIdentity(name=long_name),
                             info=CameraInfo("video 1", long_name, kind="capture"))
        self.assertLessEqual(len(camera_reports.browser_url(huge)), camera_reports.MAX_URL)

    def test_save_report_overwrites(self):
        """The report is saved once per key under output/logs.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(paths, "OUTPUT_DIR", tmp):
            first = camera_reports.save_report(webcam_report())
            second = camera_reports.save_report(webcam_report(details=(("New", "1"),)))
            self.assertEqual(first, second)
            self.assertEqual(os.path.dirname(first), os.path.join(tmp, "logs"))
            self.assertEqual(os.path.basename(first),
                             "camera-report-Webcam-C270-046d-0825.md")
            with open(first, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("| New | 1 |", text)
            self.assertTrue(text.startswith("# Camera report: Webcam C270"))
            odd = camera_reports.save_report(camera_reports.Report("///", "t", "b", "s"))
            self.assertEqual(os.path.basename(odd), "camera-report-device.md")


class GitHubTests(unittest.TestCase):
    """Finding and filing reports through the GitHub API."""

    def setUp(self):
        """Point at a fake repository.

        Args: None.
        Returns: None.
        """
        patcher = mock.patch.dict(os.environ, {feedback.REPO_ENV: "owner/repo"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_find_report_ignores_prs_and_lookalikes(self):
        """Only an issue for exactly this key counts as already reported.

        Args: None.
        Returns: None.
        """
        key = "Webcam C270 (046d:0825)"
        listing = [
            {"title": "Camera report: " + key, "body": "", "pull_request": {},
             "html_url": "pr"},
            {"title": "Camera report: " + key + " 2", "body": "", "html_url": "other"},
            {"title": "renamed", "body": "text\n" + camera_reports.marker(key),
             "html_url": "real"},
        ]
        with mock.patch("urllib.request.urlopen", return_value=reply(listing)) as urlopen:
            found = camera_reports.find_report(key, "secret")
        self.assertEqual(found["html_url"], "real")
        sent = urlopen.call_args[0][0]
        self.assertEqual(sent.get_method(), "GET")
        self.assertIn("/repos/owner/repo/issues?", sent.full_url)
        self.assertIn("labels=camera-support", sent.full_url)
        self.assertIn("state=all", sent.full_url)
        by_title = [{"title": "Camera report: " + key, "body": None, "html_url": "t"}]
        with mock.patch("urllib.request.urlopen", return_value=reply(by_title)):
            self.assertEqual(camera_reports.find_report(key, "secret")["html_url"], "t")
        with mock.patch("urllib.request.urlopen", return_value=reply([])):
            self.assertIsNone(camera_reports.find_report(key, "secret"))

    def test_anonymous_request_has_no_auth_header(self):
        """Without a token the request is anonymous (the repository is public).

        Args: None.
        Returns: None.
        """
        with mock.patch("urllib.request.urlopen", return_value=reply([])) as urlopen:
            feedback.github_request("GET", "/repos/owner/repo/issues", "")
        sent = urlopen.call_args[0][0]
        self.assertIsNone(sent.get_header("Authorization"))
        self.assertIsNone(sent.data)

    def test_file_report_posts_label(self):
        """Filing POSTs the title, body and camera-support label with the token.

        Args: None.
        Returns: None.
        """
        report = webcam_report()
        created = {"html_url": "https://github.com/owner/repo/issues/9"}
        with mock.patch("urllib.request.urlopen", return_value=reply(created)) as urlopen:
            issue = camera_reports.file_report(report, "secret")
        self.assertEqual(issue, created)
        sent = urlopen.call_args[0][0]
        self.assertEqual(sent.get_method(), "POST")
        self.assertEqual(sent.get_header("Authorization"), "Bearer secret")
        payload = json.loads(sent.data)
        self.assertEqual(payload, {"title": report.title, "body": report.body,
                                   "labels": ["camera-support"]})


class EnvironmentLinesTests(unittest.TestCase):
    """Build and platform lines shared by feature requests and reports."""

    def test_environment_lines(self):
        """Two bullets: the build, then the platform.

        Args: None.
        Returns: None.
        """
        lines = feedback.environment_lines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("- Build: "))
        self.assertTrue(lines[1].startswith("- Platform: "))


if __name__ == "__main__":
    unittest.main()
