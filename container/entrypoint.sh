#!/bin/sh
# Start a virtual X display, then hand over to the sidecar, which launches
# the game and serves frames and key input over TCP.
set -eu

DISPLAY_NUM="${DISPLAY_NUM:-99}"
# Defaults mirror container/sidecar/config.py
GAME_WIDTH="${GAME_WIDTH:-480}"
GAME_HEIGHT="${GAME_HEIGHT:-270}"
# The window must be fully on screen to be captured. Leave a margin for
# wherever the game places it, and never go below a floor in case the game
# refuses a small window and opens a larger one.
SCREEN_MARGIN=64
SCREEN_WIDTH=$((GAME_WIDTH + SCREEN_MARGIN))
SCREEN_HEIGHT=$((GAME_HEIGHT + SCREEN_MARGIN))
[ "${SCREEN_WIDTH}" -ge 1024 ] || SCREEN_WIDTH=1024
[ "${SCREEN_HEIGHT}" -ge 640 ] || SCREEN_HEIGHT=640
SCREEN_GEOMETRY="${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24"
export DISPLAY=":${DISPLAY_NUM}" GAME_WIDTH GAME_HEIGHT

rm -f "/tmp/.X${DISPLAY_NUM}-lock"
Xvfb "${DISPLAY}" -screen 0 "${SCREEN_GEOMETRY}" -nolisten tcp &
XVFB_PID=$!

# Wait for the X socket instead of sleeping a fixed time
tries=0
until [ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; do
    tries=$((tries + 1))
    if [ "${tries}" -gt 50 ] || ! kill -0 "${XVFB_PID}" 2>/dev/null; then
        echo "entrypoint: Xvfb did not come up on ${DISPLAY}" >&2
        exit 1
    fi
    sleep 0.1
done

cd /opt/app
exec python3 -m container.sidecar
