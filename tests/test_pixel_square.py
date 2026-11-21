"""Camera-free checks for the pixel square: placement, values and the engine.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import time
import unittest

import numpy as np

from espi_live.engine import Engine
from espi_live.pixel_square import (DEFAULT_SQUARE_SIZE, SQUARE_INTERVAL_S,
                                    SQUARE_SIZES, SquarePacket, square_area,
                                    square_values)
from espi_live.processing import ESPIProcessor
from list_log import ListLog


def wait_until(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class PlacementTests(unittest.TestCase):
    """square_area and square_values."""

    def test_sizes_are_powers_of_two_with_a_default_among_them(self):
        """The choices run 1, 2, 4 … 64 and include the default.

        Args: None.
        Returns: None.
        """
        self.assertEqual(SQUARE_SIZES, tuple(2 ** n for n in range(7)))
        self.assertIn(DEFAULT_SQUARE_SIZE, SQUARE_SIZES)

    def test_default_center_is_the_image_center(self):
        """With no center the square sits in the middle.

        Args: None.
        Returns: None.
        """
        self.assertEqual(square_area(None, 8, (48, 64)), (28, 20, 8))
        self.assertEqual(square_area(None, 1, (48, 64)), (32, 24, 1))

    def test_square_moves_to_stay_inside_and_shrinks_only_for_tiny_images(self):
        """A square clicked at an edge is moved in, not cut.

        Args: None.
        Returns: None.
        """
        self.assertEqual(square_area((0, 0), 8, (48, 64)), (0, 0, 8))
        self.assertEqual(square_area((63, 47), 8, (48, 64)), (56, 40, 8))
        self.assertEqual(square_area((5, 5), 64, (10, 20)), (0, 0, 10))

    def test_values_are_cut_in_displayed_orientation(self):
        """Flipped, the square holds the mirrored pixels, mirrored back.

        Args: None.
        Returns: None.
        """
        image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
        plain = square_values(image, 1, 2, 2)
        np.testing.assert_array_equal(plain, [[17, 18], [25, 26]])
        flipped = image[:, ::-1]  # what the display shows when flipped
        np.testing.assert_array_equal(square_values(image, 1, 2, 2, flip=True),
                                      flipped[2:4, 1:3])
        plain[0, 0] = 0
        self.assertEqual(image[2, 1], 17)  # a copy, not a view

    def test_summary_and_box(self):
        """The summary names the place and the min, mean and max.

        Args: None.
        Returns: None.
        """
        values = np.array([[0, 1], [2, 7]], np.uint8)
        packet = SquarePacket(values, "Reference", 10, 20, 2, False)
        self.assertEqual(packet.box(), (10, 20, 12, 22))
        self.assertEqual(packet.summary(), "2×2 at x 10–11, y 20–21 · difference "
                         "min 0, mean 2.50, max 7")
        raw = SquarePacket(values, "Raw", 10, 20, 2, False)
        self.assertIn("level min 0", raw.summary())
        waiting = SquarePacket(None, "Reference", 10, 20, 2, False)
        self.assertTrue(waiting.summary().endswith("no reference yet (press F)"))
        earlier = SquarePacket(None, "Previous", 10, 20, 2, False)
        self.assertTrue(earlier.summary().endswith("waiting for an earlier frame"))


class EngineSquareTests(unittest.TestCase):
    """The engine cuts the square only while asked."""

    def setUp(self):
        """Create a demo engine (not started).

        Args: None.
        Returns: None.
        """
        self.engine = Engine(ListLog())
        self.engine.use_demo()
        self.addCleanup(self.stop)

    def stop(self):
        """Stop the engine's worker.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.engine.wait(5.0)

    def test_nothing_is_cut_unless_asked(self):
        """Without a request there is no square.

        Args: None.
        Returns: None.
        """
        self.engine._cut_square(np.zeros((48, 64), np.uint8), True, "Raw", False, 1.0)
        self.assertIsNone(self.engine.latest_square())

    def test_cut_at_once_then_once_per_interval(self):
        """A new request is cut on the next frame; then every interval.

        Args: None.
        Returns: None.
        """
        engine = self.engine
        engine.set_square(2, (5, 5))
        image = np.zeros((48, 64), np.uint8)
        image[4, 4] = 9
        engine._cut_square(image, True, "Reference", False, 10.0)
        first = engine.latest_square()
        np.testing.assert_array_equal(first.values, [[9, 0], [0, 0]])
        self.assertEqual((first.left, first.top, first.view, first.demo),
                         (4, 4, "Reference", True))
        engine._cut_square(image, True, "Reference", False, 10.0 + SQUARE_INTERVAL_S / 2)
        self.assertIs(engine.latest_square(), first)
        engine._cut_square(image, False, "Reference", False, 10.0 + 2 * SQUARE_INTERVAL_S)
        self.assertIsNone(engine.latest_square().values)
        engine.set_square(4, (5, 5))  # a change is cut at once
        self.assertIsNone(engine.latest_square())
        engine._cut_square(image, True, "Reference", False, 10.0 + 2 * SQUARE_INTERVAL_S)
        self.assertEqual(engine.latest_square().size, 4)
        engine.set_square(None)
        engine._cut_square(image, True, "Reference", False, 20.0)
        self.assertIsNone(engine.latest_square())

    def test_values_are_the_raw_absolute_difference(self):
        """A 3-level change reads 3; unchanged pixels read 0.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        base = np.full((48, 64), 100, np.uint8)
        moved = base.copy()
        moved[10, 10] = 103
        processor.set_reference(base, 0.0)
        self.engine.set_square(2, (11, 11))
        self.engine._cut_square(processor.difference(moved, "Reference"), True,
                                "Reference", False, 0.0)
        np.testing.assert_array_equal(self.engine.latest_square().values,
                                      [[3, 0], [0, 0]])

    def test_running_engine_publishes_raw_reference_and_previous_squares(self):
        """Raw gives levels; Reference waits for F, then gives differences.

        Args: None.
        Returns: None.
        """
        engine = self.engine
        engine.update_settings(mode="Raw")
        engine.set_square(8)
        engine.start()

        def newest(view, ready):
            """Wait for a square of a view.

            Args: view is the Settings mode; ready is whether it has values.
            Returns: True when the newest square matches.
            """
            packet = engine.latest_square()
            return (packet is not None and packet.view == view and
                    (packet.values is not None) == ready)

        self.assertTrue(wait_until(lambda: newest("Raw", True)))
        self.assertEqual(engine.latest_square().values.shape, (8, 8))
        engine.update_settings(mode="Reference")
        self.assertTrue(wait_until(lambda: newest("Reference", False)))
        engine.request("reference")
        self.assertTrue(wait_until(lambda: newest("Reference", True)))
        engine.update_settings(mode="Previous")
        self.assertTrue(wait_until(lambda: newest("Previous", True)))
        engine.update_settings(average_frames_first=True, average=2)
        self.assertTrue(wait_until(lambda: newest("Previous", True)))
        engine.update_settings(raw_difference=True)
        self.assertTrue(wait_until(lambda: newest("Raw difference", True)))


if __name__ == "__main__":
    unittest.main()
