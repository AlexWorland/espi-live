# iPhone camera over USB

An iPhone can be a camera source for ESPI Live. The **ESPI Camera** iOS app
(`ios/ESPICamera/`, installed through TestFlight) captures the back camera
with every automatic adjustment locked and serves each frame's 8-bit luma
plane, uncompressed, over the USB cable. ESPI Live reads it like any other
camera (`src/espi_live/iphone.py`), and every setting can be changed from
ESPI Live, so nobody has to switch to the phone during a measurement.

## Using it

1. **Windows, once:** install **iTunes from apple.com** (not the Microsoft
   Store version). It installs the *Apple Mobile Device Service*, which carries
   the connection over USB. The Store's iTunes and *Apple Devices* apps do not
   reliably provide it. Check with `sc query "Apple Mobile Device Service"`.
   macOS has it built in.
2. **Phone, once:** install **TestFlight** from the App Store, accept the
   ESPI-Live-Camera invitation, and install the app.
3. Plug the iPhone into the PC, unlock it, and tap **Trust** when it asks about
   the computer.
4. Open **ESPI Camera** on the phone and keep it on screen. iOS stops the
   camera when the app is in the background or the phone locks.
5. Start ESPI Live. The phone appears in the camera list as
   **iPhone (USB) (iphone-xxxxxxxx)**. With no FLIR camera or capture adapter,
   ESPI Live opens it at startup, and **Switch to live capture** finds it from
   demo mode.

### Controls in ESPI Live

| ESPI Live control | Phone setting |
|---|---|
| Exposure (µs) | Exposure duration (custom exposure; never automatic) |
| Gain (dB) | ISO: 0 dB is the phone's lowest ISO, +20 dB is ten times that |
| iPhone > Frame rate, or Requested FPS | Capture frame rate |
| iPhone > Frame size | Capture format (unbinned formats preferred) |
| iPhone > Crop | Square centre crop sent at full resolution |
| iPhone > Send at most | Cap on frames sent per second |
| iPhone > Focus | Locked lens position (0 near, 1 far) |
| iPhone > Autofocus once, then lock | One centre autofocus, then lock |

The **iPhone** section appears only while an iPhone is displayed. Changes
made on the phone's screen are sent back, and ESPI Live's controls follow
them within a second or two.

### Bayer RAW mode

The **Capture** choice in ESPI Live's iPhone section (and on the phone's screen) switches
from the video pipeline to **Bayer RAW**. Apple's video output is always processed (noise
reduction, sharpening, tone mapping); the photo output can deliver the sensor's Bayer data
with that pipeline bypassed, so RAW mode takes **stills** as fast as they come:

- **Rate:** about 5–10 frames per second (Apple's figure; measure it on your phone).
  Frame rate, frame size and the video crop sizes do not apply; **Send at most** and
  **Crop** (in plane pixels) still do.
- **Resolution:** Apple offers Bayer RAW only binned to about 12 MP (4032 × 3024); full
  resolution is processed ProRAW, which is not used. One colour plane is half that size in
  each direction (2016 × 1512).
- **Colour:** the phone keeps the sites of one colour (**RAW colour**): red for a 633 nm
  laser, green for 532 nm (both green sites averaged), or blue. Interpolated colours are
  never mixed in, so each value is a real sensor reading.
- **Values:** 16-bit counts, with the black and white level from the photo's DNG metadata
  (if the phone reports none, the app uses 0 and 16383 and says so in its error line).
  ESPI Live converts them to **deep frames**: float32 grey levels on the 8-bit scale
  (black = 0, white = 255), negatives kept, so every processing step keeps its precision
  and only the final picture is rounded to 8 bits (`processing.to_uint8`).
- **Bandwidth:** a red plane is about 6 MB per frame, so 5–10 fps needs about 30–60 MB/s:
  a USB 3 phone and cable, or a crop on USB 2.
- **Limits:** the phone still has a rolling shutter, and stills arrive with uneven timing,
  so RAW suits a continuous laser and not pulsed or phase-stepped capture. A shutter sound
  may play where the phone's region requires it.

### Bandwidth

Frames are never compressed. When USB is still busy with the previous frame,
the phone skips a frame rather than queue or compress it, and the status line
shows how many it skipped.

| Model | USB | Practical |
|---|---|---|
| Lightning iPhones, non-Pro USB-C iPhones | USB 2 | about 30 MB/s: 1920 × 1080 at about 15 fps, or a 1024 × 1024 crop at 30 fps |
| iPhone 15 Pro and later Pro models with a USB 3 cable | USB 3 | 1920 × 1080 at 30 fps and more |

1920 × 1080 Mono8 is 2.07 MB a frame (62 MB/s at 30 fps).

### What the app locks

- **Exposure:** custom duration and ISO.
- **White balance:** unity gains, so luma is a fixed mix of the sensor's red,
  green and blue response.
- **Focus:** one autofocus, then locked.
- **Off:** HDR, low-light boost, geometric distortion correction,
  stabilization and (iOS 27) low-light video noise reduction.
- **Tone:** global tone mapping where the format supports it, which is one
  curve per frame instead of local adjustments.

### Limits

- **Rolling shutter:** about 5 ms readout. It suits a continuous laser and a
  static or slowly moving object. It does not suit pulsed lasers or
  phase-stepped capture.
- **Colour filter:** a 633 nm laser lights mainly the red pixels (a quarter of
  the sensor), and 532 nm mainly the green (half). Luma interpolates the rest.
- **Built-in noise reduction:** the phone's own spatial and temporal noise
  reduction cannot be turned off on the video path.
- **Speckle size:** the lens is fixed (f/1.78 on recent main cameras), which
  gives small speckle compared with the 1.2–2.4 µm pixels.

Treat the phone as a qualitative fringe viewer and check results against the
FLIR camera.

## How the link works

- usbmuxd (Apple Mobile Device Service on Windows, built into macOS) carries
  TCP connections over the USB cable.
- The app listens on TCP port **47820** of the phone's loopback interface,
  with no Wi-Fi involved.
- ESPI Live asks usbmuxd to connect to that port (`src/espi_live/usbmux.py`,
  standard library only):
  1. Connect to `127.0.0.1:27015` on Windows, or `/var/run/usbmuxd`
     elsewhere. `USBMUXD_SOCKET_ADDRESS` overrides this.
  2. Send `ListDevices` on one socket.
  3. Send `Connect` with `DeviceID` and the byte-swapped port on another. On
     success that socket becomes the raw link.
  4. Each daemon message is a 16-byte little-endian header (length, version 1,
     type 8, tag) followed by an XML plist.
- Apple does not support this technique for third-party apps, but it is what
  iTunes, Xcode and many tools use.

### Link protocol, version 1

Every message is a 16-byte header, then the payload. All integers are
little-endian.

| Bytes | Field |
|---|---|
| 0–3 | magic `ESPC` |
| 4–5 | version (1) |
| 6–7 | kind |
| 8–11 | payload length |
| 12–15 | reserved (0) |

| Kind | Direction | Payload |
|---|---|---|
| 1 HELLO | phone → PC, on connect | JSON: `protocol_version`, `app_version`, `device_model`, `system_version`, `camera`, `formats` (`width`, `height`, `min_fps`, `max_fps`, `binned`), `limits`, `settings` |
| 2 STATUS | phone → PC, every second and after each change | JSON: `settings`, `limits`, `thermal`, `interrupted`, `camera_fps`, `sent_fps`, `sent_frames`, `dropped_frames`, `focus_locked`, `error` |
| 3 FRAME | phone → PC | 48-byte FRAME_META, then `stride × height` Mono8 bytes |
| 4 COMMAND | PC → phone | JSON with any of `exposure_us`, `iso`, `fps`, `lens_position`, `width` + `height`, `crop_width` + `crop_height`, `max_send_fps`, `refocus` |
| 5 ERROR | phone → PC | JSON `message` |

**FRAME_META** (`<IIIIQqfffI`): width, height, stride, pixel format (1 =
Mono8), sequence, presentation time (ns), exposure (µs), ISO, lens position,
reserved.

**RAW frames** (pixel format 2 = Mono16): after FRAME_META comes a 16-byte
**RAW_META** (`<ffII`): black level, white level (sensor counts), colour (0 red,
1 green, 2 blue) and the sensor's pattern (0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR); then
`stride × height` little-endian 16-bit values, one colour plane, rows unpadded
(`stride` counts values). `settings` also carries `raw` (bool) and `raw_channel`
(`"red"`, `"green"` or `"blue"`), and COMMAND accepts the same two keys. An older
app ignores them, and an older ESPI Live rejects a Mono16 frame as "not Mono8".

**settings:** `exposure_us`, `iso`, `fps`, `lens_position`, `width`,
`height`, `crop_width`, `crop_height`, `max_send_fps`.

**limits:** `[low, high]` pairs for `exposure_us`, `iso`, `fps` and
`lens_position`.

If no message arrives for 5 s, ESPI Live treats the link as lost and
reconnects every 2 s.

Keep `ios/ESPICamera/Sources/LinkProtocol.swift` and
`src/espi_live/iphone.py` in step. Both have tests that pin the byte layout
(`ESPICameraTests`, `tests/test_iphone.py`).

## Building and shipping the app

See `ios/README.md`.
