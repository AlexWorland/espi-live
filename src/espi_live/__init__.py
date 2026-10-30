"""Live FLIR Blackfly viewer for ESPI (electronic speckle pattern interferometry).

Run with ``python -m espi_live [--demo]`` and ``src/`` on PYTHONPATH (``run.sh``
and ``scripts/windows/SETUP AND START ESPI.bat`` do this for you).
"""

import logging

# Nothing is printed for library-style use; AppLog attaches the real handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())
