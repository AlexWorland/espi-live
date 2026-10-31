"""Camera-free checks for the bundled HTML user guide and text fallback.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import espi_live.guide as guide_module
from espi_live.guide import GUIDE_FILENAME, guide_html, guide_path


class GuideTest(unittest.TestCase):
    """Check that the manual exists, is complete, and links correctly."""

    def test_guide_file_exists(self):
        """The guide ships next to the package, ready for PyInstaller.

        Args: None.
        Returns: None.
        """
        path = guide_path()
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(os.path.basename(path), GUIDE_FILENAME)
        self.assertTrue(GUIDE_FILENAME.endswith(".html"))

    def test_guide_is_self_contained_html(self):
        """The page has inline styles and scripts and needs no network.

        Args: None.
        Returns: None.
        """
        text = guide_html()
        self.assertTrue(text.lstrip().startswith("<!doctype html>"))
        self.assertIn("<title>ESPI live — User Guide</title>", text)
        self.assertIn("<style>", text)
        self.assertIn("<script>", text)
        self.assertIsNone(re.search(r'(?:src|href)="https?://', text))

    def test_guide_covers_every_control_and_operation(self):
        """The guide documents every visible control and common operation.

        Args: None.
        Returns: None.
        """
        text = guide_html()
        self.assertGreater(len(text), 20000)
        for phrase in ("Raw", "Previous", "Reference", "Exposure", "Gain",
                       "Requested FPS", "Average N", "Stretch to 0–255",
                       "Palette", "Gamma", "CLAHE", "Record AVI",
                       "Save frame", "Set reference", "Reset last",
                       "Reset average", "Switch to live capture",
                       "Fit to window", "Actual size", "F1"):
            self.assertTrue(phrase in text, f"missing phrase: {phrase!r}")
        for operation in ("Capture a reference", "Record a measurement",
                          "Save a still frame", "Troubleshooting",
                          "Defaults and limits", "Packaged builds",
                          "Waiting for camera frame"):
            self.assertTrue(operation in text, f"missing section: {operation!r}")
        self.assertTrue("below 1 <strong>brighten</strong>" in text)
        self.assertTrue("Other changes do not split the file" in text)
        self.assertTrue("0–50,000 µs" in text)
        self.assertTrue("59 fps" in text)

    def test_guide_matches_the_combined_camera_and_help_features(self):
        """The guide describes the restart fix, presets and rotating help tips.

        Args: None.
        Returns: None.
        """
        text = guide_html()
        for phrase in ("Restarting a camera", "current exposure, gain and FPS",
                       "Live preview", "Measurement", "Which settings should I use?",
                       "Did you know", "Help tips", "display gamma",
                       "Send error reports automatically"):
            self.assertIn(phrase, text)
        self.assertNotIn("8000 µs, 0 dB and 30 fps every time", text)
        self.assertNotIn("re-enter your exposure", text)
        self.assertNotIn("&lt; 1 darkens", text)

    def test_internal_links_have_targets(self):
        """Every in-page link points at an id that exists on the page.

        Args: None.
        Returns: None.
        """
        text = guide_html()
        targets = set(re.findall(r'\bid="([^"]+)"', text))
        links = set(re.findall(r'href="#([^"]+)"', text))
        missing = sorted(links - targets)
        self.assertEqual(missing, [], f"links without targets: {missing}")

    def test_targets_are_unique(self):
        """No id appears twice, so every link resolves to one place.

        Args: None.
        Returns: None.
        """
        ids = re.findall(r'\bid="([^"]+)"', guide_html())
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        self.assertEqual(duplicates, [], f"duplicate ids: {duplicates}")

    def test_print_overrides_dark_theme(self):
        """Print rules beat the dark-theme variables and set light colors.

        Args: None.
        Returns: None.
        """
        text = guide_html()
        block = text[text.index("@media print"):text.index("</style>")]
        self.assertIn(":root[data-theme=\"dark\"]", block)
        self.assertIn("color-scheme: light", block)
        self.assertIn("--fg: #000", block)
        self.assertIn("--bg: #fff", block)

    def test_frozen_guide_path(self):
        """A PyInstaller-style _MEIPASS folder is searched for the guide.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            guide_dir = os.path.join(folder, "espi_live")
            os.makedirs(guide_dir)
            target = os.path.join(guide_dir, GUIDE_FILENAME)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("<!doctype html>")
            with patch.object(sys, "_MEIPASS", folder, create=True):
                self.assertEqual(guide_path(), target)

    def test_macos_app_resources_are_searched(self):
        """A .app bundle finds the guide in Contents/Resources.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            contents = os.path.join(folder, "Contents")
            target = os.path.join(contents, "Resources", "espi_live",
                                  GUIDE_FILENAME)
            os.makedirs(os.path.dirname(target))
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("<!doctype html>")
            executable = os.path.join(contents, "MacOS", "ESPI Live")
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "executable", executable), \
                    patch.object(sys, "_MEIPASS",
                                 os.path.join(contents, "Frameworks"),
                                 create=True):
                self.assertEqual(guide_path(), target)

    def test_missing_guide_falls_back_to_the_source_path(self):
        """With no frozen layout and no file, the expected source path returns.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            expected = os.path.join(folder, GUIDE_FILENAME)
            with patch.object(guide_module, "__file__",
                              os.path.join(folder, "guide.py")), \
                    patch.object(sys, "_MEIPASS", folder, create=True):
                self.assertEqual(guide_path(), expected)

    def test_unreadable_guide_returns_an_html_error(self):
        """A read failure returns a valid HTML page naming the path.

        Args: None.
        Returns: None.
        """
        guide_html.cache_clear()
        with patch("builtins.open", side_effect=OSError("no access")):
            text = guide_html()
        guide_html.cache_clear()
        self.assertIn("User guide unavailable", text)
        self.assertIn("no access", text)
        self.assertTrue(text.startswith("<!doctype html>"))

if __name__ == "__main__":
    unittest.main()
