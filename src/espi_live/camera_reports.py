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

import os
import re
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
    Returns: An HTML comment string.
    """
    return f"<!-- espi-live camera report: {key} -->"


def cell(text):
    """Make text safe inside a Markdown table cell.

    Args: text is any value.
    Returns: The text with pipes and backticks escaped and line breaks
        turned into spaces.
    """
    text = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|").replace("`", "\\`")


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
    facts = [f"- {line}" for line in description[1:]]
    facts += [f"- ⚠ {note}" for note in identity.warnings()]
    head = ["The viewer was switched to a video device it does not recognise yet. "
            "Everything it could read is below; add the device to "
            "`capture_models.ADAPTERS` if it should be supported.", "",
            f"**{description[0]}**", *facts, "", "### What the device reports", "",
            "| Item | Value |", "| --- | --- |"]
    rows = [f"| {cell(label)} | {cell(value)} |" for label, value in details]
    tail = ["", "### Where it was seen", "", *feedback.environment_lines(), "",
            "_Filed automatically by ESPI live when the device was switched on._",
            "", marker(key)]
    summary = "\n".join([f"**{description[0]}**", *facts, "",
                         "_The full report is saved on the PC under output/logs/; "
                         "paste it below._", "", marker(key)])
    return Report(key=key, title=report_title(key),
                  body=fit(head, rows, tail, MAX_BODY), summary=summary)


def browser_url(report):
    """Build a prefilled new-issue page for copies without a token.

    Args: report is a Report.
    Returns: An https URL of at most MAX_URL characters; the summary is
        shortened if needed (titles are already short).
    """
    body = report.summary

    def build():
        """Encode the page address for the current body.

        Args: None.
        Returns: The URL.
        """
        query = urllib.parse.urlencode({"title": report.title, "body": body,
                                        "labels": LABEL})
        return f"https://github.com/{feedback.repository()}/issues/new?{query}"

    url = build()
    while len(url) > MAX_URL:
        body = body[:len(body) * 3 // 4]
        url = build()
    return url


def save_report(report):
    """Write the full report under output/logs, replacing an earlier copy.

    Args: report is a Report.
    Returns: The file path.
    """
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", report.key).strip("-.") or "device"
    path = os.path.join(paths.output_folder("logs"), f"camera-report-{slug}.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {report.title}\n\n{report.body}\n")
    return path


def find_report(key, token):
    """Look for an issue that already reports this key.

    Args: key is the report key; token authorizes the call ("" reads the
        public repository anonymously).
    Returns: The issue dict, or None. Raises feedback.FeedbackError.
    """
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
