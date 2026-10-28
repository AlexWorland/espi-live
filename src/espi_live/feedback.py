"""Send feature requests from the app to GitHub as issues.

A request becomes an issue labelled ``feature-request`` in the repository the
build came from. With "enrich" ticked it also gets the ``enrich`` label, which
starts the ``enrich-feature-request.yml`` workflow: GitHub Copilot rewrites
the request into a structured spec and appends it to the issue, so the app
itself holds no AI credentials.

Builds embed a token from the ``FEEDBACK_TOKEN`` secret (a fine-grained token
with only Issues: read & write on this repository); ``feedback-token.txt``
beside the program or ``ESPI_FEEDBACK_TOKEN`` overrides it. Without a token
the request can still be opened in a browser as a prefilled new-issue page.
No GUI imports: the front ends drive this module.
"""

import json
import os
import platform
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from espi_live import updater

DEFAULT_REPO = "AlexWorland/espi-live"  # source checkouts have no baked-in REPO
REPO_ENV = "ESPI_FEEDBACK_REPO"
TOKEN_ENV = "ESPI_FEEDBACK_TOKEN"
TOKEN_FILE = "feedback-token.txt"
REQUEST_LABEL = "feature-request"
ENRICH_LABEL = "enrich"
MAX_TITLE = 200


class FeedbackError(Exception):
    """A feature request that could not be filed."""


@dataclass(frozen=True)
class FeatureRequest:
    """What the user typed, plus where it came from."""

    title: str
    description: str
    enrich: bool = False
    submitter: str = ""
    source: str = ""  # "camera" or "dummy mode"; empty when unknown


def repository():
    """Name the repository that receives requests.

    Args: None.
    Returns: "owner/name" from the environment, the build, or DEFAULT_REPO.
    """
    return (os.environ.get(REPO_ENV, "").strip()
            or updater.build_value("REPO", "") or DEFAULT_REPO)


def feedback_token():
    """Choose the token that may create issues.

    Args: None.
    Returns: The token string, or "" when this copy cannot file directly.
    """
    return updater.read_token(TOKEN_ENV, TOKEN_FILE, "FEEDBACK_TOKEN")


def validate(request):
    """Check that a request is complete enough to file.

    Args: request is a FeatureRequest.
    Returns: None. Raises FeedbackError naming the missing part.
    """
    if not request.title.strip():
        raise FeedbackError("enter a short title")
    if not request.description.strip():
        raise FeedbackError("describe the feature you would like")


def context_lines(request):
    """List the facts about this copy that help a developer act on a request.

    Args: request is the FeatureRequest.
    Returns: A list of Markdown bullet strings.
    """
    lines = [f"- Build: {updater.describe_build()}",
             f"- Platform: {platform.system()} {platform.release()} "
             f"({platform.machine()}), Python {platform.python_version()}"]
    if request.source:
        lines.append(f"- Running with: {request.source}")
    if request.submitter.strip():
        lines.append(f"- Requested by: {request.submitter.strip()}")
    return lines


def issue_payload(request):
    """Build the GitHub "create issue" body for a request.

    Args: request is a validated FeatureRequest.
    Returns: A dict with title, body and labels.
    """
    labels = [REQUEST_LABEL] + ([ENRICH_LABEL] if request.enrich else [])
    body = "\n".join([request.description.strip(), "", "---",
                      "_Sent from ESPI live's **Request feature** button._", "",
                      *context_lines(request)])
    return {"title": request.title.strip()[:MAX_TITLE], "body": body,
            "labels": labels}


def browser_url(request):
    """Build a prefilled new-issue page for filing without a token.

    Args: request is a FeatureRequest.
    Returns: The https URL; labels apply only if the signed-in user may set them.
    """
    payload = issue_payload(request)
    query = urllib.parse.urlencode({"title": payload["title"], "body": payload["body"],
                                    "labels": ",".join(payload["labels"])})
    return f"https://github.com/{repository()}/issues/new?{query}"


def explain_http_error(exc):
    """Turn a GitHub HTTP or network error into a FeedbackError message.

    Args: exc is the urllib error.
    Returns: A FeedbackError to raise.
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return FeedbackError("the feedback token was rejected (expired or revoked); "
                                 f"put a new one in {TOKEN_FILE} beside the program")
        if exc.code in (403, 404):
            return FeedbackError("the feedback token may not create issues in "
                                 f"{repository()} (needs Issues: read & write)")
        if exc.code == 410:
            return FeedbackError(f"issues are turned off in {repository()}")
        return FeedbackError(f"GitHub returned HTTP {exc.code}")
    return FeedbackError(f"cannot reach GitHub: {getattr(exc, 'reason', exc)}")


def submit(request):
    """File a feature request as a GitHub issue.

    Args: request is a FeatureRequest.
    Returns: The new issue's web URL. Raises FeedbackError on failure.
    """
    validate(request)
    token = feedback_token()
    if not token:
        raise FeedbackError("this copy has no feedback token")
    data = json.dumps(issue_payload(request)).encode("utf-8")
    http = urllib.request.Request(
        f"{updater.API_ROOT}/repos/{repository()}/issues", data=data, method="POST",
        headers={"Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "espi-live-feedback",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(http, timeout=updater.TIMEOUT_S,
                                    context=updater.ssl_context()) as response:
            issue = json.load(response)
    except (urllib.error.URLError, OSError) as exc:
        raise explain_http_error(exc) from exc
    except ValueError as exc:
        raise FeedbackError(f"unreadable reply from GitHub: {exc}") from exc
    return issue["html_url"]


class FeedbackWorker:
    """File requests on a background thread for a front end.

    The front end polls ``events`` for ("sent", (request, url)) and
    ("failed", (request, message)); the request is returned so a failure can
    fall back to the browser without losing the text.
    """

    def __init__(self):
        """Create an idle worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.busy = False

    def submit(self, request):
        """Start filing a request.

        Args: request is a FeatureRequest.
        Returns: False if a request is already being sent, else True.
        """
        if self.busy:
            return False
        self.busy = True

        def work():
            """Thread body: file the request and post its event.

            Args: None.
            Returns: None.
            """
            try:
                self.events.put(("sent", (request, submit(request))))
            except Exception as exc:
                self.events.put(("failed", (request, str(exc))))
            finally:
                self.busy = False

        threading.Thread(target=work, name="espi-feedback", daemon=True).start()
        return True
