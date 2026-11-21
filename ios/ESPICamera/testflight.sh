#!/usr/bin/env bash
# Archive the ESPI Camera app and upload it to App Store Connect (TestFlight).
# Needs a Mac with Xcode 26 or later and XcodeGen (brew install xcodegen).
#
#   DEVELOPMENT_TEAM=ABCDE12345 ASC_KEY_ID=KEYID ASC_ISSUER_ID=uuid \
#     ios/ESPICamera/testflight.sh [build-number]
#
# ASC_KEY_ID / ASC_ISSUER_ID are an App Store Connect API key (Users and
# Access > Integrations > App Store Connect API, Admin or App Manager role);
# the key file is read from ASC_KEY_PATH, default
# ~/.appstoreconnect/private_keys/AuthKey_<ASC_KEY_ID>.p8. Never put it in the
# repository. Without them, Xcode's signed-in Apple ID is used instead.
#
# The build number defaults to a UTC timestamp (YYMMDDHHMM), which always
# increases, so every upload is accepted. Output goes to build/testflight/
# next to this script (git-ignored). See ios/README.md for the one-time setup.
set -euo pipefail
cd "$(dirname "$0")"

: "${DEVELOPMENT_TEAM:?Set DEVELOPMENT_TEAM to your Apple Developer Team ID}"
build_number="${1:-$(date -u +%y%m%d%H%M)}"

auth=()
if [[ -n "${ASC_KEY_ID:-}" || -n "${ASC_ISSUER_ID:-}" ]]; then
  : "${ASC_KEY_ID:?Set ASC_KEY_ID together with ASC_ISSUER_ID}"
  : "${ASC_ISSUER_ID:?Set ASC_ISSUER_ID together with ASC_KEY_ID}"
  key_path="${ASC_KEY_PATH:-$HOME/.appstoreconnect/private_keys/AuthKey_${ASC_KEY_ID}.p8}"
  [[ -f "$key_path" ]] || { echo "API key not found at $key_path" >&2; exit 1; }
  auth=(-authenticationKeyPath "$key_path" -authenticationKeyID "$ASC_KEY_ID"
        -authenticationKeyIssuerID "$ASC_ISSUER_ID")
fi

out=build/testflight
rm -rf "$out"
mkdir -p "$out"

cat > "$out/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key><string>app-store-connect</string>
	<key>destination</key><string>upload</string>
	<key>teamID</key><string>${DEVELOPMENT_TEAM}</string>
	<key>signingStyle</key><string>automatic</string>
	<key>uploadSymbols</key><true/>
	<key>manageAppVersionAndBuildNumber</key><false/>
</dict>
</plist>
PLIST

echo "==> Generating the Xcode project"
xcodegen generate --quiet

echo "==> Archiving build ${build_number}"
xcodebuild archive \
  -project ESPICamera.xcodeproj -scheme ESPICamera \
  -destination 'generic/platform=iOS' \
  -archivePath "$out/ESPICamera.xcarchive" \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" CURRENT_PROJECT_VERSION="$build_number" \
  -allowProvisioningUpdates ${auth[@]+"${auth[@]}"}

echo "==> Exporting and uploading to App Store Connect"
xcodebuild -exportArchive \
  -archivePath "$out/ESPICamera.xcarchive" \
  -exportOptionsPlist "$out/ExportOptions.plist" \
  -exportPath "$out/export" \
  -allowProvisioningUpdates ${auth[@]+"${auth[@]}"}

echo "==> Uploaded build ${build_number}. It appears in TestFlight after processing (10 to 30 min)."
