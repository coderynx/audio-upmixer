#!/bin/sh
set -eu
web_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "${1:-}" = --capture ]; then
  bundle="${TMPDIR:-/tmp}/upmixer-native-audio-test.app"
  mkdir -p "$bundle/Contents/MacOS"
  binary="$bundle/Contents/MacOS/audio-test"
  cat > "$bundle/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>com.coderynx.upmixer.audio-test</string>
<key>CFBundleName</key><string>Upmixer Audio Test</string>
<key>CFBundleExecutable</key><string>audio-test</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>Verify Upmixer playback by recording the BlackHole loopback device. This test does not select your physical microphone.</string>
</dict></plist>
PLIST
else
  binary=$(mktemp -t upmixer-native-audio)
  trap 'rm -f "$binary"' EXIT
fi
xcrun clang -fobjc-arc -Wall -Wextra -Werror -mmacosx-version-min=15.0 \
  "$web_dir/src-tauri/native/tests/audio_bridge_test.m" \
  -framework AVFoundation -framework AVFAudio -framework Foundation \
  -framework CoreMedia -framework AudioToolbox -framework CoreAudio -o "$binary"
if [ "${1:-}" = --capture ]; then
  codesign --force --sign - "$bundle"
  log=$(mktemp -t upmixer-audio-capture)
  trap 'rm -f "$log"' EXIT
  open -n -W --stdout "$log" --stderr "$log" "$bundle" --args --capture
  cat "$log"
  # Launch Services does not propagate the app's exit status.
  grep -q 'PASS: capture and transport checks complete' "$log"
else
  "$binary" "$@"
fi
