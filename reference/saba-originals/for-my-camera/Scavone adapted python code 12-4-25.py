
"""
blackfly_espi_easypyspin.py

Simple ESPI-style live view with a Teledyne FLIR Blackfly camera.

Keys:
  q : quit
  r : grab new reference frame (for reference differencing)
  m : toggle mode (consecutive vs reference difference)
"""

import cv2
import EasyPySpin
import numpy as np


def rescale_to_8bit(img: np.ndarray) -> np.ndarray:
    """Stretch float32 image to 0–255 uint8 for display."""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - min_val) / (max_val - min_val) * 255.0
    return scaled.astype(np.uint8)


def main():
    # Open first Blackfly on the bus
    cap = EasyPySpin.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open Blackfly via EasyPySpin")

    # Try to force mono, smallish ROI and manual exposure to reduce bandwidth/lag
    # (properties are silently ignored if unsupported)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)          # keep as mono if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)       # adjust to your camera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Turn off auto exposure/gain if supported
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # OpenCV convention (may or may not work)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)            # camera-specific; tune as needed

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to grab initial frame from Blackfly")

    # Convert first frame to float32 grayscale
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = gray.astype(np.float32)
    # DEBUG VISIBILITY BOOST
    debug = np.clip(gray * 5.0, 0, 255).astype(np.uint8)
    cv2.imshow("DEBUG RAW BOOST", debug)

    print("min/max raw:", gray.min(), gray.max())

    ref_frame = gray.copy()   # reference for reference-diff mode
    prev_frame = gray.copy()  # previous frame for consecutive-diff mode

    use_consecutive = True  # True: |I(t)-I(t-1)|; False: |I(t)-I_ref|

    print("Running ESPI demo.")
    print("Keys: q=quit, r=new reference frame, m=toggle mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, stopping.")
            break

        # Convert to grayscale float32
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray_f = gray.astype(np.float32)

        if use_consecutive:
            diff = cv2.absdiff(gray_f, prev_frame)
            prev_frame[:] = gray_f
        else:
            diff = cv2.absdiff(gray_f, ref_frame)

        diff_8u = rescale_to_8bit(diff)

        # Optionally apply slight blur to reduce speckle noise
        diff_8u = cv2.GaussianBlur(diff_8u, (3, 3), 0)

        cv2.imshow("Blackfly RAW", gray.astype(np.uint8))
        cv2.imshow(
            f"ESPI diff ({'consecutive' if use_consecutive else 'reference'})",
            diff_8u,
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            ref_frame[:] = gray_f
            print("New reference frame captured.")
        elif key == ord("m"):
            use_consecutive = not use_consecutive
            print("Mode:",
                  "consecutive |I(t)-I(t-1)|"
                  if use_consecutive
                  else "reference |I(t)-I_ref|")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
