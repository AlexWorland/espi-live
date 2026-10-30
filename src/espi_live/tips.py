"""Short help tips for the status bar.

``TIPS`` is the whole list: adding one is a new line. ``TipRotator`` walks
that list and remembers a dismissal. It does not import Qt or touch the
engine; the window owns the timer and runs it on the UI thread.
"""

import random

# Long enough to say one thing. The window is sized so each one fits on the
# status bar without squeezing the dummy-mode badge; a longer entry fails.
MAX_TIP_LENGTH = 46
# Inside the 10–15 s the tips should wait. Slow, so the image stays the focus.
ROTATE_MS = 12_000
PREFIX = "Did you know: "

TIPS = (
    "Did you know: hover a control for help",
    "Did you know: S saves and F sets a reference",
    "Did you know: stretch clip sets contrast",
    "Did you know: gamma bends display brightness",
    "Did you know: a rolling average updates live",
    "Did you know: rolling off waits for N frames",
    "Did you know: Request feature sends an idea",
    "Did you know: Log and Controls can float",
    "Did you know: the camera list switches view",
    "Did you know: scroll to zoom and drag to pan",
)


class TipRotator:
    """Which tip is showing, and whether tips are hidden."""

    def __init__(self, tips=TIPS, index=None, hidden=False):
        """Start on one tip.

        Args: tips is the list to walk (the shared ``TIPS`` unless a test
            passes its own); index selects one (None picks at random, so
            repeat visits do not always start on the first); hidden starts
            dismissed.
        Returns: None.
        """
        if not tips:
            raise ValueError("tips must not be empty")
        self.tips = tuple(tips)
        if index is None:
            index = random.randrange(len(self.tips))
        self.index = index % len(self.tips)
        self.hidden = bool(hidden)

    def current(self):
        """Return the tip on screen.

        Args: None.
        Returns: The current tip, or "" when tips are hidden.
        """
        if self.hidden:
            return ""
        return self.tips[self.index]

    def advance(self):
        """Move to the next tip, wrapping at the end.

        Args: None.
        Returns: The tip now current ("" when hidden). The position moves
            either way.
        """
        self.index = (self.index + 1) % len(self.tips)
        return self.current()

    def dismiss(self):
        """Hide tips until restore.

        Args: None.
        Returns: None.
        """
        self.hidden = True

    def restore(self):
        """Show tips again, on the tip last selected.

        Args: None.
        Returns: None.
        """
        self.hidden = False
