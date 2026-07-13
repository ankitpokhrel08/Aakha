#!/usr/bin/env bash
# Rebuild the Aakha Android debug APK. Run from anywhere.
# Toolchain was installed via Homebrew (openjdk@21 + android-commandlinetools).
set -euo pipefail

export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
export PATH="$JAVA_HOME/bin:$PATH"

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

npx cap sync android                       # copy www/ + config into the android project
printf 'sdk.dir=%s\n' "$ANDROID_HOME" > android/local.properties
( cd android && ./gradlew :app:assembleDebug --no-daemon )

mkdir -p dist
cp android/app/build/outputs/apk/debug/app-debug.apk dist/Aakha-debug.apk
echo "APK -> $DIR/dist/Aakha-debug.apk"
