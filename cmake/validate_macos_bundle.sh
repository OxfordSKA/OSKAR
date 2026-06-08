#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <package.dmg>" >&2
    exit 2
fi

dmg_path="$1"
if [ ! -f "$dmg_path" ]; then
    echo "DMG not found: $dmg_path" >&2
    exit 2
fi

mount_dir="$(mktemp -d)"
mach_o_list="$(mktemp)"
attached=0

cleanup()
{
    if [ "$attached" -eq 1 ]; then
        hdiutil detach -force "$mount_dir" >/dev/null
    fi
    rm -rf "$mount_dir" "$mach_o_list"
}
trap cleanup EXIT

printf 'Y\n' | hdiutil attach -nobrowse -readonly \
    -mountpoint "$mount_dir" "$dmg_path" >/dev/null
attached=1

app_path="$(find "$mount_dir" -maxdepth 2 -type d -name '*.app' -print -quit)"
if [ -z "$app_path" ]; then
    echo "No application bundle found in $dmg_path" >&2
    exit 1
fi

while IFS= read -r -d '' file_path; do
    if file -b "$file_path" | grep -q 'Mach-O'; then
        printf '%s\0' "$file_path" >> "$mach_o_list"
    fi
done < <(find "$app_path" -type f -print0)

if [ ! -s "$mach_o_list" ]; then
    echo "No Mach-O files found in $app_path" >&2
    exit 1
fi

checked=0
while IFS= read -r -d '' file_path; do
    relative_path="${file_path#"$app_path"/}"
    architectures="$(lipo -archs "$file_path")"
    if [ "$architectures" != "arm64" ]; then
        echo "Unexpected architecture in $relative_path: $architectures" >&2
        exit 1
    fi

    min_os="$(vtool -show-build "$file_path" 2>/dev/null |
        awk '$1 == "minos" {print $2; exit}')"
    if [ -z "$min_os" ]; then
        echo "Could not determine deployment target for $relative_path" >&2
        exit 1
    fi
    if ! awk -v version="$min_os" 'BEGIN {
        split(version, actual, ".");
        minor = (actual[2] == "") ? 0 : actual[2] + 0;
        exit !((actual[1] < 11) || (actual[1] == 11 && minor == 0))
    }'; then
        echo "Deployment target exceeds macOS 11.0 in $relative_path: $min_os" >&2
        exit 1
    fi

    if otool -L "$file_path" |
            grep -E '(^|[[:space:]])(/opt/homebrew|/usr/local|/Users)/' \
            >/dev/null; then
        echo "Non-bundled dependency found in $relative_path:" >&2
        otool -L "$file_path" |
            grep -E '(^|[[:space:]])(/opt/homebrew|/usr/local|/Users)/' >&2
        exit 1
    fi

    codesign --verify --strict "$file_path"
    checked=$((checked + 1))
done < "$mach_o_list"

codesign --verify --deep --strict "$app_path"

echo "Validated $checked arm64 Mach-O files in $(basename "$app_path")"
