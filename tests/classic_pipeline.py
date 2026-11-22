"""The processing many engine checks were written for: no default noise pipeline.

The default Settings average camera frames before the difference, average
16 frames into an automatic reference and run the Oriented speckle filter.
Checks of one mechanism (one-frame references, difference averaging, the
speckle filter widths) turn those off with CLASSIC_PIPELINE, so they test
exactly what they name; test_noise_pipeline checks the defaults themselves.
"""

from espi_live.engine import Settings

CLASSIC_PIPELINE = {
    "average": 1,
    "average_frames_first": False,
    "ref_frames": 1,
    "smoothing": 0,
    "smoothing_method": "Gaussian",
    "auto_reference": False,
}


def classic_settings(**changes):
    """Build Settings without the default noise pipeline.

    Args: changes are further Settings fields to set.
    Returns: A Settings.
    """
    return Settings(**{**CLASSIC_PIPELINE, **changes})
