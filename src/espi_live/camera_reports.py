"""Report unrecognised cameras to GitHub so support can be added.

When someone switches on a video device the viewer does not recognise, the
engine builds a Report from everything the device tells OpenCV. This module
formats it, files it as an issue labelled ``camera-support`` with the
Issues-only feedback token (see feedback.py), and never files the same
device type twice: before filing it looks for an existing report by a hidden
marker or its exact title. Copies without a token save the report under
output/logs/ and offer a prefilled new-issue page instead. No GUI imports;
network calls run only on the background worker the front end polls.
"""

import html
import os
import queue
import re
import threading
import urllib.parse
from dataclasses import dataclass

from espi_live import feedback, paths

LABEL = "camera-support"
TITLE_PREFIX = "Camera report: "
MAX_BODY = 60000  # GitHub's issue body limit is 65 536 characters
MAX_URL = 8000    # browsers and GitHub reject much longer prefilled links


@dataclass(frozen=True)
class Report:
    """One camera report: its de-duplication key, issue text and short form."""

    key: str
    title: str
    body: str
    summary: str  # short Markdown for a prefilled browser page


def report_title(key):
    """Title the issue for a report key.

    Args: key is the report key.
    Returns: "Camera report: <key>", cut to GitHub-friendly length.
    """
    return (TITLE_PREFIX + key)[:feedback.MAX_TITLE]


def marker(key):
    """Build the hidden marker that identifies a report in an issue body.

    Args: key is the report key.
    Returns: An HTML comment string; every run of two or more dashes in the
        key is split so a device name cannot end the comment early and
        inject text after it.
    """
    return f"<!-- espi-live camera report: {re.sub(r'-{2,}', '- -', key)} -->"


def cell(text):
    """Make text safe inside a Markdown table cell.

    Args: text is any value.
    Returns: The text with pipes, backticks and brackets escaped and line
        breaks turned into spaces.
    """
    text = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|").replace("`", "\\`")
    return text.replace("[", "\\[").replace("]", "\\]")


def prose(text):
    """Make one line of report prose safe inside Markdown.

    Args: text is any value, such as a device name the report quotes.
    Returns: The text with the characters Markdown treats as structure
        backslash-escaped and line breaks turned into spaces.
    """
    text = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    for character in "\\`*_[]":
        text = text.replace(character, "\\" + character)
    return text


def fit(head, rows, tail, limit):
    """Join a report, dropping table rows from the end until it fits.

    Args: head and tail are lists of lines kept whole; rows are the table
        rows between them; limit is the maximum length in characters.
    Returns: The joined text, with a "N more rows not shown" row when rows
        were dropped.
    """
    kept = list(rows)
    text = "\n".join(head + kept + tail)
    while len(text) > limit and kept:
        cut = max(1, len(kept) // 10)
        kept = kept[:-cut]
        note = [f"| … | {len(rows) - len(kept)} more rows not shown |"]
        text = "\n".join(head + kept + note + tail)
    return text


def capture_report(info, identity, details):
    """Build the report for an unrecognised video device.

    Args: info is the device's CameraInfo; identity its CaptureIdentity;
        details the (label, text) pairs from CaptureSession.read_details().
    Returns: A Report keyed by device name and USB ID.
    """
    name = identity.name or info.model
    if identity.vid is not None and identity.pid is not None:
        key = f"{name or 'Video device'} ({identity.vid:04x}:{identity.pid:04x})"
    else:
        key = name or info.serial
    description = identity.description()
    facts = [f"- {prose(line)}" for line in description[1:]]
    facts += [f"- ⚠ {prose(note)}" for note in identity.warnings()]
    head = ["The viewer was switched to a video device it does not recognise yet. "
            "Everything it could read is below; add the device to "
            "`capture_models.ADAPTERS` if it should be supported.", "",
            f"**{prose(description[0])}**", *facts, "", "### What the device reports", "",
            "| Item | Value |", "| --- | --- |"]
    rows = [f"| {cell(label)} | {cell(value)} |" for label, value in details]
    tail = ["", "### Where it was seen", "", *feedback.environment_lines(), "",
            "_Filed automatically by ESPI live when the device was switched on._",
            "", marker(key)]
    summary = "\n".join([f"**{prose(description[0])}**", *facts, "",
                         "_The full report is saved on the PC under output/logs/; "
                         "paste it below._", "", marker(key)])
    return Report(key=key, title=report_title(key),
                  body=fit(head, rows, tail, MAX_BODY), summary=summary)


def browser_url(report):
    """Build a prefilled new-issue page for copies without a token.

    Args: report is a Report.
    Returns: An https URL of at most MAX_URL characters; the summary is
        shortened if needed (titles are already short), and characters that
        cannot appear in a URL-encoded string (such as lone surrogates in a
        device name) are dropped instead of raising.
    """
    body = report.summary

    def build():
        """Encode the page address for the current body.

        Args: None.
        Returns: The URL.
        """
        query = urllib.parse.urlencode({"title": report.title, "body": body,
                                        "labels": LABEL}, errors="ignore")
        return f"https://github.com/{feedback.repository()}/issues/new?{query}"

    url = build()
    while len(url) > MAX_URL:
        cut = body[:len(body) * 3 // 4]
        body = cut.encode("utf-8", "ignore").decode("utf-8", "ignore")
        url = build()
    return url


def save_report(report):
    """Write the full report under output/logs, replacing an earlier copy.

    Args: report is a Report.
    Returns: The file path.
    """
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", report.key).strip("-.") or "device"
    path = os.path.join(paths.output_folder("logs"), f"camera-report-{slug}.md")
    with open(path, "w", encoding="utf-8", errors="ignore") as handle:
        handle.write(f"# {report.title}\n\n{report.body}\n")
    return path


def find_report(key, token):
    """Look for an issue that already reports this key.

    Args: key is the report key; token authorizes the call ("" reads the
        public repository anonymously).
    Returns: The issue dict, or None. Raises feedback.FeedbackError.
    """
    # per_page=100 reads the newest page only; ample for years of camera reports.
    query = urllib.parse.urlencode({"labels": LABEL, "state": "all", "per_page": 100})
    issues = feedback.github_request(
        "GET", f"/repos/{feedback.repository()}/issues?{query}", token)
    title, hidden = report_title(key), marker(key)
    for issue in issues:
        if "pull_request" in issue:
            continue
        if hidden in (issue.get("body") or "") or issue.get("title") == title:
            return issue
    return None


def file_report(report, token):
    """File a report as a new issue.

    Args: report is a Report; token may create issues.
    Returns: The created issue dict. Raises feedback.FeedbackError.
    """
    return feedback.github_request(
        "POST", f"/repos/{feedback.repository()}/issues", token,
        {"title": report.title, "body": report.body, "labels": [LABEL]})


@dataclass(frozen=True)
class ReportState:
    """Where a report stands: filed or found, saved locally, or failed."""

    url: str = ""          # the issue that holds the report
    created: bool = False  # this run filed it
    path: str = ""         # local copy saved for a copy without a token
    error: str = ""        # why it could not be filed


def resolve(report):
    """Find or file one report, or save it when this copy has no token.

    Args: report is a Report.
    Returns: A ReportState. Raises only for failures outside GitHub, such as
        a report that cannot be saved.
    """
    token = feedback.feedback_token()
    try:
        issue = find_report(report.key, token)
        if issue is not None:
            return ReportState(url=issue["html_url"])
        if token:
            return ReportState(url=file_report(report, token)["html_url"], created=True)
        return ReportState(path=save_report(report))
    except feedback.FeedbackError as exc:
        return ReportState(error=str(exc), path="" if token else save_report(report))


class CameraReportWorker:
    """File camera reports on one background thread for a front end.

    The front end polls ``events`` for ("state", (key, ReportState)), one per
    submitted report, in submission order.
    """

    def __init__(self):
        """Create an idle worker; its thread starts with the first report.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self._jobs = queue.Queue()
        self._seen = set()  # keys submitted this run (front-end thread only)
        self._thread = None

    def submit(self, report):
        """Queue a report unless its key was already submitted this run.

        Args: report is a Report.
        Returns: True when queued, False for a repeat.
        """
        if report.key in self._seen:
            return False
        self._seen.add(report.key)
        self._jobs.put(report)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="espi-camera-report",
                                            daemon=True)
            self._thread.start()
        return True

    def _run(self):
        """Thread body: resolve queued reports one at a time, forever.

        Args: None.
        Returns: None (the daemon thread ends with the program).
        """
        while True:
            report = self._jobs.get()
            try:
                state = resolve(report)
            except Exception as exc:
                state = ReportState(error=f"{type(exc).__name__}: {exc}")
            self.events.put(("state", (report.key, state)))


def status_html(state):
    """Describe a report's state for the camera information panel.

    Args: state is a ReportState.
    Returns: Rich text with the issue link where there is one.
    """
    if state.url:
        link = (f'<a href="{html.escape(state.url, quote=True)}">'
                f"{html.escape(state.url)}</a>")
        return (f"Reported to GitHub as {link}." if state.created
                else f"Already reported as {link}.")
    if state.error:
        return f"Report not sent: {html.escape(state.error)}."
    if state.path:
        return ("This copy cannot file reports on GitHub; the report is saved to "
                f"{html.escape(state.path)}.")
    return "Reporting this camera to GitHub…"


def needs_button(state):
    """Decide whether to offer filing the report in a browser.

    Args: state is a ReportState.
    Returns: True when no issue exists and filing failed or is impossible.
    """
    return not state.url and bool(state.error or state.path)
