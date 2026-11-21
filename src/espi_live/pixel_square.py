"""The pixel square: an N×N patch of the raw difference, value by value.

The Pixels window shows a small square of the image enlarged, with every
pixel's integer value written on it. The values are the raw absolute
difference in camera levels (0–255), taken before the noise threshold,
Difference gain, averaging and Scaling, so a still scene reads 0 (or a few
levels of sensor noise) and movement raises them. In Raw view they are the
camera levels.

The square is placed on the displayed image (which Flip horizontal may
mirror), so square_values cuts it from the frame in displayed orientation.

Nothing here touches a GUI toolkit or PySpin.
"""

from dataclasses import dataclass

import numpy as np

SQUARE_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_SQUARE_SIZE = 8
# The engine cuts a new square this often while the Pixels window is open:
# readable, and cheap next to a 30 fps camera.
SQUARE_INTERVAL_S = 0.1


def square_area(center, size, shape):
    """Place an N×N square on the displayed image, inside it.

    Args: center is the (x, y) displayed pixel the square is centered on, or
        None for the image center; size is N; shape is the image's (height,
        width).
    Returns: (left, top, size) in displayed pixels. The square is moved, not
        cut, to stay inside the image, and shrinks only when the image is
        smaller than N.
    """
    height, width = shape[:2]
    size = max(1, min(int(size), width, height))
    x, y = center if center is not None else (width // 2, height // 2)
    left = min(max(int(x) - size // 2, 0), width - size)
    top = min(max(int(y) - size // 2, 0), height - size)
    return left, top, size


def square_values(image, left, top, size, flip=False):
    """Cut the square's pixels from a frame, as they appear on screen.

    Args: image is the frame (raw difference, or the frame in Raw view);
        left, top and size place the square in displayed pixels; flip is True
        when the display mirrors the frame left-right.
    Returns: A size×size uint8 copy in displayed orientation.
    """
    if flip:
        right = image.shape[1] - left
        return image[top:top + size, right - size:right][:, ::-1].copy()
    return image[top:top + size, left:left + size].copy()


@dataclass(frozen=True)
class SquarePacket:
    """One published pixel square.

    values is the size×size uint8 array in displayed orientation, or None
    while there is no baseline to subtract (no reference or earlier frame
    yet); view is the engine's view ("Raw", "Previous" or "Reference");
    left, top and size place it on the displayed image; demo marks
    synthetic video.
    """

    values: object
    view: str
    left: int
    top: int
    size: int
    demo: bool

    def box(self):
        """Return the square as a box on the displayed image.

        Args: None.
        Returns: (left, top, right, bottom), right and bottom exclusive.
        """
        return self.left, self.top, self.left + self.size, self.top + self.size

    def summary(self):
        """Describe where the square is and what it holds.

        Args: None.
        Returns: Text such as "8×8 at x 476–483, y 316–323 · difference
            min 0, mean 0.41, max 7".
        """
        right, bottom = self.left + self.size - 1, self.top + self.size - 1
        where = (f"{self.size}×{self.size} at x {self.left}–{right}, "
                 f"y {self.top}–{bottom}")
        if self.values is None:
            waiting = ("no reference yet (press F)" if self.view == "Reference"
                       else "waiting for an earlier frame")
            return f"{where} · {waiting}"
        kind = "level" if self.view == "Raw" else "difference"
        values = np.asarray(self.values)
        return (f"{where} · {kind} min {int(values.min())}, "
                f"mean {float(values.mean()):.2f}, max {int(values.max())}")
