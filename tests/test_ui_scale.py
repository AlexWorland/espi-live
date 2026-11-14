"""Camera-free checks for the interface-size arithmetic (no Qt involved).

Run from the project root:
    PYTHONPATH=src python -m unittest tests.test_ui_scale
"""

import unittest

from espi_live import ui_scale


class PercentTests(unittest.TestCase):
    """Allowed percentages, stored values and the presets."""

    def tearDown(self):
        """Put the shared size back to the default.

        Args: None.
        Returns: None.
        """
        ui_scale.set_percent(ui_scale.DEFAULT_PERCENT)

    def test_the_default_is_the_standard_size_and_inside_the_range(self):
        """100% is today's look, and the range brackets it with the presets inside it.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.DEFAULT_PERCENT, 100)
        self.assertLess(ui_scale.MIN_PERCENT, ui_scale.DEFAULT_PERCENT)
        self.assertGreater(ui_scale.MAX_PERCENT, ui_scale.DEFAULT_PERCENT)
        self.assertEqual(ui_scale.PRESETS[0], ui_scale.MIN_PERCENT)
        self.assertEqual(ui_scale.PRESETS[-1], ui_scale.MAX_PERCENT)
        self.assertIn(ui_scale.DEFAULT_PERCENT, ui_scale.PRESETS)
        self.assertEqual(list(ui_scale.PRESETS), sorted(set(ui_scale.PRESETS)))

    def test_clamp_rounds_and_limits(self):
        """Out-of-range numbers are pulled in and fractions rounded.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.clamp_percent(1), ui_scale.MIN_PERCENT)
        self.assertEqual(ui_scale.clamp_percent(5000), ui_scale.MAX_PERCENT)
        self.assertEqual(ui_scale.clamp_percent(124.6), 125)
        self.assertEqual(ui_scale.clamp_percent(100), 100)

    def test_parse_reads_whatever_a_settings_file_stores(self):
        """Numbers, text and junk all give an allowed percentage.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.parse_percent(125), 125)
        self.assertEqual(ui_scale.parse_percent("150"), 150)
        self.assertEqual(ui_scale.parse_percent("133.0"), 133)
        self.assertEqual(ui_scale.parse_percent("999"), ui_scale.MAX_PERCENT)
        for junk in (None, "", "big", [], float("nan"), float("inf")):
            self.assertEqual(ui_scale.parse_percent(junk), ui_scale.DEFAULT_PERCENT, junk)

    def test_larger_and_smaller_walk_the_presets(self):
        """Larger and Smaller move to the next preset and stop at the ends.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.larger(100), 110)
        self.assertEqual(ui_scale.larger(103), 110)  # a slider value between presets
        self.assertEqual(ui_scale.smaller(100), 90)
        self.assertEqual(ui_scale.smaller(103), 100)
        self.assertEqual(ui_scale.larger(ui_scale.MAX_PERCENT), ui_scale.MAX_PERCENT)
        self.assertEqual(ui_scale.smaller(ui_scale.MIN_PERCENT), ui_scale.MIN_PERCENT)

    def test_the_shared_size_is_clamped_and_read_back(self):
        """set_percent stores an allowed value that percent(), factor() and px() use.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.set_percent(150), 150)
        self.assertEqual(ui_scale.percent(), 150)
        self.assertEqual(ui_scale.factor(), 1.5)
        self.assertEqual(ui_scale.px(130), 195)
        self.assertEqual(ui_scale.set_percent(10_000), ui_scale.MAX_PERCENT)
        self.assertEqual(ui_scale.set_percent(0), ui_scale.MIN_PERCENT)


class PixelTests(unittest.TestCase):
    """Scaling pixel lengths and style sheets."""

    def test_scaled_rounds_to_whole_pixels_and_never_loses_a_line(self):
        """A positive length stays at least one pixel; zero stays zero.

        Args: None.
        Returns: None.
        """
        self.assertEqual(ui_scale.scaled(130, 100), 130)
        self.assertEqual(ui_scale.scaled(130, 125), 163)  # 162.5 rounds up
        self.assertEqual(ui_scale.scaled(104, 200), 208)
        self.assertEqual(ui_scale.scaled(1, 70), 1)
        self.assertEqual(ui_scale.scaled(0, 200), 0)
        self.assertEqual(ui_scale.scaled(-3, 200), 0)

    def test_style_sheet_lengths_scale_but_hairlines_do_not(self):
        """Paddings and radii follow the size; a 1px border stays crisp.

        Args: None.
        Returns: None.
        """
        style = "border:1px solid red;padding:2px 8px;border-radius:4px;color:#163a6b;"
        self.assertEqual(ui_scale.scale_stylesheet(style, 100), style)
        self.assertEqual(
            ui_scale.scale_stylesheet(style, 150),
            "border:1px solid red;padding:3px 12px;border-radius:6px;color:#163a6b;")
        self.assertEqual(ui_scale.scale_stylesheet("margin-top:8.5px;", 200), "margin-top:17px;")
        self.assertEqual(ui_scale.scale_stylesheet("font-weight:bold;", 200), "font-weight:bold;")


if __name__ == "__main__":
    unittest.main()
