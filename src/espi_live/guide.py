"""Locate and read the bundled HTML user guide.

The manual is one self-contained HTML page (``user_guide.html``) with inline
styles and scripts, so it opens offline in the default browser with working
section links, a contents filter, collapsible troubleshooting answers and a
print view. The Help menu opens it in the default browser and reports the
path when no browser can be started.
"""

import html
import os
import sys
from functools import lru_cache

GUIDE_FILENAME = "user_guide.html"


def guide_path():
    """Locate the bundled user guide from source or a packaged build.

    Args: None.
    Returns: The absolute path of the HTML file. The path is returned even
        when the file is missing, so callers can report where it was expected.
    """
    source_root = os.path.dirname(os.path.abspath(__file__))
    roots = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(frozen_root)
    if getattr(sys, "frozen", False):
        executable_dir = os.path.dirname(os.path.abspath(sys.executable))
        contents_dir = os.path.dirname(executable_dir)
        roots.extend((executable_dir, contents_dir,
                      os.path.join(contents_dir, "Resources")))
    roots.append(source_root)
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        for candidate in (os.path.join(root, "espi_live", GUIDE_FILENAME),
                          os.path.join(root, GUIDE_FILENAME)):
            if os.path.isfile(candidate):
                return candidate
    return os.path.join(source_root, GUIDE_FILENAME)


@lru_cache(maxsize=1)
def guide_html():
    """Read the user guide once and cache it.

    Args: None.
    Returns: The HTML source. If the file cannot be read, a small valid HTML
        page names the expected path and the error.
    """
    path = guide_path()
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>ESPI live — User guide unavailable</title></head>"
                "<body><h1>User guide unavailable</h1>"
                f"<p>The guide could not be read from <code>{html.escape(path)}"
                f"</code>.</p><p>Error: {html.escape(str(exc))}</p>"
                "<p>Reinstall the application or run it from the project "
                "folder.</p></body></html>\n")
