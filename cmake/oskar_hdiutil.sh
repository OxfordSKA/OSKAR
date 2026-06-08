#!/bin/bash
#
# Wrapper around hdiutil used by CPack's Bundle generator.
#
# Two robustness fixes over a bare "hdiutil $@":
#   1. "detach" gets -force, so a busy volume (Spotlight/fsevents still holding
#      the freshly written disk image) is ejected instead of hanging forever.
#   2. Retries are bounded, so a genuinely failing command cannot loop forever.

args=("$@")
if [ "${args[0]}" = "detach" ]; then
    # Insert -force immediately after the "detach" verb.
    args=("detach" "-force" "${args[@]:1}")
fi

attempt=0
max_attempts=10
hdiutil "${args[@]}"
status=$?
while [ $status -ne 0 ] && [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    sleep 1
    hdiutil "${args[@]}"
    status=$?
done
exit $status
