"""Interface size: the allowed percentages, the pixel arithmetic and the shared value.

Qt-free, so the front end and the tests share it. The window (``qt_app``)
applies the percentage to the application font, to the style's metrics and to
every pixel size it fixes itself; it asks this module for the numbers.

The current percentage is one process-wide value (there is one application
font), read through ``percent()``/``factor()``/``px()`` and written only by
``set_percent()``.
"""

import re

MIN_PERCENT = 70
MAX_PERCENT = 200
DEFAULT_PERCENT = 100
STEP_PERCENT = 5  # slider granularity
# Where View > Interface size > Larger/Smaller stops, in order.
PRESETS = (70, 80, 90, 100, 110, 125, 150, 175, 200)
SETTING_KEY = "ui/scale_percent"  # QSettings key holding the chosen percentage

# A length in a style sheet. Hairlines (1px borders) are never scaled.
_PIXELS = re.compile(r"(\d+(?:\.\d+)?)px")

_percent = DEFAULT_PERCENT


def clamp_percent(value):
    """Round a percentage and keep it inside the allowed range.

    Args: value is a number.
    Returns: An int from MIN_PERCENT to MAX_PERCENT.
    """
    return max(MIN_PERCENT, min(MAX_PERCENT, int(round(value))))


def parse_percent(value):
    """Read a stored percentage, which QSettings may hand back as text.

    Args: value is whatever was stored: a number, text such as "125", or None.
    Returns: An int inside the allowed range; DEFAULT_PERCENT when the value
        is missing or not a number.
    """
    try:
        return clamp_percent(float(value))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_PERCENT


def larger(percent):
    """Return the next preset above a percentage.

    Args: percent is the current percentage.
    Returns: The next larger entry of PRESETS, or MAX_PERCENT at the top.
    """
    return next((preset for preset in PRESETS if preset > percent), MAX_PERCENT)


def smaller(percent):
    """Return the next preset below a percentage.

    Args: percent is the current percentage.
    Returns: The next smaller entry of PRESETS, or MIN_PERCENT at the bottom.
    """
    return next((preset for preset in reversed(PRESETS) if preset < percent), MIN_PERCENT)


def scaled(value, percent):
    """Scale a pixel length.

    Args: value is the length at 100%; percent is the interface size.
    Returns: The length in whole pixels, at least 1 for a positive value.
    """
    if value <= 0:
        return 0
    return max(1, int(value * percent / 100 + 0.5))


def scale_stylesheet(style, percent):
    """Scale every pixel length in a style sheet except 1px hairlines.

    Args: style is the style sheet text written for 100%; percent is the
        interface size.
    Returns: The style sheet text.
    """
    def replace(match):
        """Scale one matched length.

        Args: match is the regular-expression match.
        Returns: The replacement text, such as "12px".
        """
        value = float(match.group(1))
        return match.group(0) if value <= 1 else f"{scaled(value, percent)}px"

    return _PIXELS.sub(replace, style)


def percent():
    """Return the interface size in force.

    Args: None.
    Returns: An int from MIN_PERCENT to MAX_PERCENT.
    """
    return _percent


def factor():
    """Return the interface size as a multiplier.

    Args: None.
    Returns: 1.0 at 100%.
    """
    return _percent / 100


def set_percent(value):
    """Set the interface size in force (the window applies it to Qt).

    Args: value is the wanted percentage; it is clamped to the allowed range.
    Returns: The percentage now in force.
    """
    global _percent
    _percent = clamp_percent(value)
    return _percent


def px(value):
    """Scale a pixel length by the interface size in force.

    Args: value is the length at 100%.
    Returns: The length in whole pixels.
    """
    return scaled(value, _percent)
