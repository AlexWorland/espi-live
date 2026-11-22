# ESPI Camera (iOS)

A SwiftUI app that streams the iPhone camera to ESPI Live over the USB cable.
How it is used and the link protocol are described in `docs/iphone/README.md`.

| Path | Contents |
|---|---|
| `ESPICamera/project.yml` | XcodeGen spec; the `.xcodeproj` is generated and git-ignored |
| `ESPICamera/Sources/CameraStreamer.swift` | Capture with locked exposure, focus and white balance; the loopback TCP listener; frame sending (one frame in flight, skip when busy) |
| `ESPICamera/Sources/LinkProtocol.swift` | Wire format shared with `src/espi_live/iphone.py` |
| `ESPICamera/Sources/ContentView.swift`, `AppModel.swift`, `CameraPreview.swift`, `ESPICameraApp.swift` | Status and controls on the phone |
| `ESPICamera/Sources/RawPipeline.swift` | The difference view's source choice (Bayer RAW stills, then 10-bit, then 8-bit luma), CFA patterns, labels, and the rules for why a still is blocked and when to fall back to luma |
| `ESPICamera/Sources/DifferenceEngine.swift`, `Difference.metal` | The GPU difference (red photosites or luma, the previous image kept on the GPU) and its on-screen drawing |
| `ESPICamera/Sources/DifferenceView.swift` | The full-screen difference view, its label and controls |
| `ESPICamera/Tests/` | Protocol tests (byte layout, parser, JSON keys); the difference view's source choice; the GPU difference checked against a CPU reference on synthetic buffers |
| `ESPICamera/testflight.sh` | Archive, sign and upload to TestFlight |

You need a Mac with Xcode 26 or later and XcodeGen (`brew install xcodegen`).
The app targets iOS 17 and the iPhone only, and is written in Swift 6 with
complete concurrency checking.

## Build and test

```sh
cd ios/ESPICamera
xcodegen generate
xcodebuild -project ESPICamera.xcodeproj -scheme ESPICamera \
  -destination 'platform=iOS Simulator,name=iPhone 17' test
```

The simulator has no camera, so the tests there cover the protocol, the
source choice and the GPU difference on synthetic images. Install
it on a phone with TestFlight, or from Xcode with your own team selected.

## TestFlight

App Store Connect app: **ESPI-Live-Camera**, bundle ID
`com.alexworland.espicamera`. The internal group **ESPI team** has access to
every build, so a processed build appears in its members' TestFlight app
with no review.

To upload a build:

```sh
DEVELOPMENT_TEAM=<team id> ASC_KEY_ID=<key id> ASC_ISSUER_ID=<issuer uuid> \
  ios/ESPICamera/testflight.sh
```

- The API key file goes in `~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8`
  with mode 600. It needs the Admin or App Manager role. Never commit it:
  `*.p8` is git-ignored.
- The build number is a UTC timestamp, so every upload is accepted.
- Processing takes 10–30 minutes.
- Builds expire after 90 days: upload again to renew.
- `ITSAppUsesNonExemptEncryption` is false, so no export-compliance question
  is asked.
- Raise `MARKETING_VERSION` in `project.yml` for a new version.

### Adding a tester

Internal testers must be members of the App Store Connect team, invited under
**Users and Access** with any role. They can be limited to this app. Once they
accept, add them to **ESPI team** in the app's TestFlight tab.
