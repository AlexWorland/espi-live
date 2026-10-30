"""Checks for the status-bar help tips. No Qt, camera or dialog."""

import unittest

from espi_live import tips


class TipListTests(unittest.TestCase):
    """The shared list stays short and non-empty."""

    def test_list_is_non_empty_and_each_tip_fits_one_line(self):
        """Every tip is a short "Did you know" sentence, and none repeat.

        Args: None.
        Returns: None.
        """
        self.assertTrue(tips.TIPS)
        self.assertEqual(len(tips.TIPS), len(set(tips.TIPS)))
        for tip in tips.TIPS:
            self.assertTrue(tip.startswith(tips.PREFIX))
            self.assertGreater(len(tip), len(tips.PREFIX))
            self.assertLessEqual(len(tip), tips.MAX_TIP_LENGTH)
            self.assertEqual(tip, tip.strip())
            self.assertNotIn("\n", tip)


class TipRotatorTests(unittest.TestCase):
    """Walking the list, wrapping, and dismissal."""

    def test_random_start_is_one_of_the_tips(self):
        """Omitting the index picks a real tip.

        Args: None.
        Returns: None.
        """
        rotator = tips.TipRotator()
        self.assertIn(rotator.current(), tips.TIPS)
        self.assertFalse(rotator.hidden)

    def test_given_index_wraps_into_range(self):
        """An index past either end lands on a real tip.

        Args: None.
        Returns: None.
        """
        self.assertEqual(tips.TipRotator(index=-1).current(), tips.TIPS[-1])
        self.assertEqual(tips.TipRotator(index=len(tips.TIPS)).index, 0)

    def test_empty_list_is_refused(self):
        """There is nothing to rotate without a tip.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(ValueError):
            tips.TipRotator(tips=())

    def test_advance_moves_to_the_next_tip_and_wraps(self):
        """The next tip follows, and the last one returns to the first.

        Args: None.
        Returns: None.
        """
        rotator = tips.TipRotator(index=0)
        self.assertEqual(rotator.advance(), tips.TIPS[1])
        self.assertEqual(rotator.current(), tips.TIPS[1])
        rotator.index = len(tips.TIPS) - 1
        self.assertEqual(rotator.advance(), tips.TIPS[0])
        self.assertEqual(rotator.index, 0)

    def test_dismiss_hides_the_text_until_restore(self):
        """Dismissal clears the text and is undone by restore.

        Args: None.
        Returns: None.
        """
        rotator = tips.TipRotator(index=0)
        rotator.dismiss()
        self.assertTrue(rotator.hidden)
        self.assertEqual(rotator.current(), "")
        self.assertEqual(rotator.advance(), "")
        self.assertEqual(rotator.index, 1)
        rotator.restore()
        self.assertFalse(rotator.hidden)
        self.assertEqual(rotator.current(), tips.TIPS[1])

    def test_hidden_flag_starts_dismissed(self):
        """hidden=True begins with no text showing.

        Args: None.
        Returns: None.
        """
        rotator = tips.TipRotator(index=2, hidden=True)
        self.assertEqual(rotator.current(), "")
        self.assertEqual(rotator.index, 2)
