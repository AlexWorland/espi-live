"""Send feature requests from the app to GitHub as issues.

A request becomes an issue labelled ``feature-request`` and ``enrich`` in the
repository the build came from. The ``enrich`` label starts
``enrich-feature-request.yml``: GitHub Copilot writes a structured spec from
the title when the description is empty, and expands the user's text when it
is not, then appends that spec to the issue. The app itself holds no AI
credentials.

Builds embed a token from the ``FEEDBACK_TOKEN`` secret (a fine-grained token
with only Issues: read & write on this repository); ``feedback-token.txt``
beside the program or ``ESPI_FEEDBACK_TOKEN`` overrides it. Without a token
the request cannot be filed. No GUI imports: the front ends drive this module.
"""

import json
import os
import platform
import queue
import threading
import urllib.error
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

    A description is optional: an empty one is filled in from the title when
    the issue is enriched.

    Args: request is a FeatureRequest.
    Returns: None. Raises FeedbackError naming the missing part.
    """
    if not request.title.strip():
        raise FeedbackError("enter a short title")


def environment_lines():
    """Describe this copy's build and platform for an issue.

    Args: None.
    Returns: Two Markdown bullet strings: Build, then Platform.
    """
    return [f"- Build: {updater.describe_build()}",
            f"- Platform: {platform.system()} {platform.release()} "
            f"({platform.machine()}), Python {platform.python_version()}"]


def context_lines(request):
    """List the facts about this copy that help a developer act on a request.

    Args: request is the FeatureRequest.
    Returns: A list of Markdown bullet strings.
    """
    lines = environment_lines()
    if request.source:
        lines.append(f"- Running with: {request.source}")
    if request.submitter.strip():
        lines.append(f"- Requested by: {request.submitter.strip()}")
    return lines


def issue_payload(request):
    """Build the GitHub "create issue" body for a request.

    Every request is labelled for enrichment. The user's description is kept
    when they wrote one; otherwise the body says so, and enrichment writes the
    spec from the title.

    Args: request is a validated FeatureRequest.
    Returns: A dict with title, body and labels.
    """
    description = request.description.strip()
    opening = description if description else "_No description was given._"
    body = "\n".join([opening, "", "---",
                      "_Sent from ESPI live's **Request feature** button._", "",
                      *context_lines(request)])
    return {"title": request.title.strip()[:MAX_TITLE], "body": body,
            "labels": [REQUEST_LABEL, ENRICH_LABEL]}


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


def github_request(method, path, token, payload=None):
    """Call the GitHub REST API.

    Args: method is "GET" or "POST"; path starts with "/repos/" and may
        carry a query string; token authorizes the call ("" for an anonymous
        call, enough to read a public repository); payload is the JSON body
        to send, if any.
    Returns: The decoded JSON reply. Raises FeedbackError on HTTP, network
        or decoding errors.
    """
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "espi-live-feedback",
               "X-GitHub-Api-Version": "2022-11-28"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    http = urllib.request.Request(f"{updater.API_ROOT}{path}", data=data,
                                  method=method, headers=headers)
    try:
        with urllib.request.urlopen(http, timeout=updater.TIMEOUT_S,
                                    context=updater.ssl_context()) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError) as exc:
        raise explain_http_error(exc) from exc
    except ValueError as exc:
        raise FeedbackError(f"unreadable reply from GitHub: {exc}") from exc


def create_issue(payload):
    """File an issue with the feedback token.

    Args: payload is the GitHub "create issue" body (title, body, labels).
    Returns: The new issue's web URL. Raises FeedbackError on failure.
    """
    token = feedback_token()
    if not token:
        raise FeedbackError("this copy has no feedback token")
    issue = github_request("POST", f"/repos/{repository()}/issues", token, payload)
    return issue["html_url"]


def submit(request):
    """File a feature request as a GitHub issue.

    Args: request is a FeatureRequest.
    Returns: The new issue's web URL. Raises FeedbackError on failure.
    """
    validate(request)
    return create_issue(issue_payload(request))


class FeedbackWorker:
    """File requests on a background thread for a front end.

    The front end polls ``events`` for ("sent", (request, url)) and
    ("failed", (request, message)); the request is returned so a failure can
    be edited and retried without losing the text. Any exception other
    than FeedbackError is a bug, so after posting the failure it is re-raised
    for threading.excepthook to log and report.
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
            except FeedbackError as exc:
                self.events.put(("failed", (request, str(exc))))
            except Exception as exc:
                self.events.put(("failed", (request, f"{type(exc).__name__}: {exc}")))
                raise
            finally:
                self.busy = False

        threading.Thread(target=work, name="espi-feedback", daemon=True).start()
        return True
