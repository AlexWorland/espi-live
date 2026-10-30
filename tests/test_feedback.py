"""Network-free checks for filing feature requests as GitHub issues.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from espi_live import feedback

REQUEST = feedback.FeatureRequest(title="  Export the reference frame ",
                                  description="Save the baseline as PNG.\n",
                                  submitter="Sam", source="camera")


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


class PayloadTests(unittest.TestCase):
    """Issue contents and validation."""

    def test_payload_has_labels_text_and_context(self):
        """A request carries both labels, the user's text and the context.

        Args: None.
        Returns: None.
        """
        payload = feedback.issue_payload(REQUEST)
        self.assertEqual(payload["title"], "Export the reference frame")
        self.assertEqual(payload["labels"], ["feature-request", "enrich"])
        self.assertTrue(payload["body"].startswith("Save the baseline as PNG."))
        self.assertIn("- Requested by: Sam", payload["body"])
        self.assertIn("- Running with: camera", payload["body"])
        self.assertIn("- Build: ", payload["body"])

    def test_blank_description_is_enriched_from_the_title(self):
        """An empty description is filed and marked so enrichment writes it.

        Args: None.
        Returns: None.
        """
        plain = feedback.FeatureRequest(title="t", description="  \n")
        payload = feedback.issue_payload(plain)
        self.assertEqual(payload["labels"], ["feature-request", "enrich"])
        self.assertTrue(payload["body"].startswith("_No description was given._"))
        self.assertNotIn("Requested by", payload["body"])
        feedback.validate(plain)

    def test_validate_rejects_a_blank_title(self):
        """A blank title is refused; a blank description is not.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(feedback.FeedbackError):
            feedback.validate(feedback.FeatureRequest(title="  ", description="d"))
        feedback.validate(feedback.FeatureRequest(title="t", description="\n "))


class SubmitTests(unittest.TestCase):
    """Token choice and the GitHub request."""

    def setUp(self):
        """Point at a fake repository with a token from the environment.

        Args: None.
        Returns: None.
        """
        patcher = mock.patch.dict(os.environ, {feedback.REPO_ENV: "owner/repo",
                                               feedback.TOKEN_ENV: "secret"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_submit_posts_issue_and_returns_url(self):
        """submit() POSTs the payload with the token and returns the issue URL.

        Args: None.
        Returns: None.
        """
        reply = json.dumps({"html_url": "https://github.com/owner/repo/issues/5"})
        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResponse(reply.encode())) as urlopen:
            url = feedback.submit(REQUEST)
        self.assertEqual(url, "https://github.com/owner/repo/issues/5")
        sent = urlopen.call_args[0][0]
        self.assertEqual(sent.get_method(), "POST")
        self.assertEqual(sent.full_url, "https://api.github.com/repos/owner/repo/issues")
        self.assertEqual(sent.get_header("Authorization"), "Bearer secret")
        self.assertEqual(json.loads(sent.data)["labels"], ["feature-request", "enrich"])

    def test_submit_without_token_fails(self):
        """A copy with no token refuses instead of posting anonymously.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {feedback.TOKEN_ENV: ""}), \
                mock.patch.object(feedback.updater, "build_value", return_value=""), \
                mock.patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(feedback.FeedbackError):
                feedback.submit(REQUEST)
        urlopen.assert_not_called()

    def test_http_errors_are_explained(self):
        """A rejected token names the file that replaces it.

        Args: None.
        Returns: None.
        """
        error = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(feedback.FeedbackError, feedback.TOKEN_FILE):
                feedback.submit(REQUEST)

    def test_token_file_beside_program_overrides_build(self):
        """feedback-token.txt beside a frozen build wins over the baked-in token.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, feedback.TOKEN_FILE), "w") as handle:
                handle.write("from-file\n")
            with mock.patch.dict(os.environ, {feedback.TOKEN_ENV: ""}), \
                    mock.patch.object(feedback.updater, "install_target",
                                      return_value=os.path.join(folder, "espi-live")):
                self.assertEqual(feedback.feedback_token(), "from-file")

    def test_other_http_and_network_errors_are_explained(self):
        """Permission, disabled-issues, other HTTP and network errors each read clearly.

        Args: None.
        Returns: None.
        """
        cases = ((403, "Issues: read & write"), (404, "Issues: read & write"),
                 (410, "turned off"), (500, "HTTP 500"))
        for code, text in cases:
            error = urllib.error.HTTPError("u", code, "x", {}, None)
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(feedback.FeedbackError, text):
                    feedback.submit(REQUEST)
        offline = urllib.error.URLError("no route")
        with mock.patch("urllib.request.urlopen", side_effect=offline):
            with self.assertRaisesRegex(feedback.FeedbackError, "cannot reach GitHub: no route"):
                feedback.submit(REQUEST)

    def test_unreadable_reply_is_reported(self):
        """A reply that is not JSON becomes a FeedbackError.

        Args: None.
        Returns: None.
        """
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"<html>")):
            with self.assertRaisesRegex(feedback.FeedbackError, "unreadable reply"):
                feedback.submit(REQUEST)


class WorkerTests(unittest.TestCase):
    """The background thread the front end polls."""

    def run_worker(self, **patch):
        """Submit REQUEST through a worker with submit() patched and wait.

        Args: patch holds mock.patch.object keyword arguments for feedback.submit.
        Returns: The (kind, payload) event the worker posted.
        """
        worker = feedback.FeedbackWorker()
        with mock.patch.object(feedback, "submit", **patch):
            self.assertTrue(worker.submit(REQUEST))
            kind, payload = worker.events.get(timeout=5)
        for _ in range(500):  # busy clears in the thread's finally, just after the event
            if not worker.busy:
                break
            time.sleep(0.01)
        self.assertFalse(worker.busy)
        return kind, payload

    def test_success_posts_the_url(self):
        """A filed request posts ("sent", (request, url)).

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.run_worker(return_value="https://x/1"),
                         ("sent", (REQUEST, "https://x/1")))

    def test_failure_posts_the_message(self):
        """A failed request posts ("failed", (request, message)).

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.run_worker(side_effect=feedback.FeedbackError("nope")),
                         ("failed", (REQUEST, "nope")))

    def test_second_submit_while_busy_is_refused(self):
        """Only one request is sent at a time.

        Args: None.
        Returns: None.
        """
        worker = feedback.FeedbackWorker()
        worker.busy = True
        self.assertFalse(worker.submit(REQUEST))
        self.assertTrue(worker.events.empty())


if __name__ == "__main__":
    unittest.main()
