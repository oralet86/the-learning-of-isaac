#!/usr/bin/env python3
"""Record Isaac gameplay via OBS Virtual Camera (V4L2), with pynput key logging."""

from datetime import datetime
import json
import os
import signal
import subprocess
import threading
import time

from pynput import keyboard

from constants import CODEC, FPS, TARGET_HEIGHT, TARGET_WIDTH

INTERVAL = 1.0 / FPS

CQ = "32"
VIDEO_FILTERS = (
    f"fps={FPS},scale={TARGET_WIDTH}:{TARGET_HEIGHT},format=gray,setpts=N/FRAME_RATE/TB"
)


def validate_video(path: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            print(f"  ffprobe failed: {result.stderr.strip()[:200]}")
            return False

        stream_info = result.stdout.strip()
        if not stream_info:
            print("  no stream info found")
            return False

        result2 = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result2.returncode != 0:
            print(f"  frame extraction failed: {result2.stderr.decode()[:200]}")
            return False

        data = result2.stdout
        if len(data) == 0:
            print("  empty frame data")
            return False

        mean_value = sum(data) / len(data)
        print(f"  stream={stream_info}  mean_pixel={mean_value:.1f}")

        if mean_value < 1.0:
            print("  BLACK SCREEN DETECTED")
            return False
        return True

    except subprocess.TimeoutExpired:
        print("  timeout during validation")
        return False
    except Exception as e:
        print(f"  validation error: {e}")
        return False


def find_v4l2_device() -> dict | None:
    """Find OBS Virtual Camera V4L2 device."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError, subprocess.TimeoutExpired:
        print("  v4l2-ctl not found. Install v4l-utils.")
        return None

    if result.returncode != 0:
        print("  v4l2-ctl failed.")
        return None

    for block in result.stdout.split("\n\n"):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        name = lines[0].strip()
        if "OBS Virtual Camera" in name or "v4l2loopback" in name:
            for line in lines[1:]:
                dev = line.strip()
                if dev.startswith("/dev/video"):
                    return {"device": dev}

    print("  No v4l2loopback / OBS Virtual Camera device found.")
    print("  Available V4L2 devices:")
    print(result.stdout[:500] if result.stdout else "  none")
    return None


def key_to_name(k):
    try:
        return k.char
    except AttributeError:
        return str(k).replace("Key.", "")


def main():
    print("Looking for OBS Virtual Camera...")
    v4l2 = find_v4l2_device()

    if v4l2 is None:
        print("OBS Virtual Camera not found.")
        print("Install v4l2loopback-dkms and start Virtual Camera in OBS.")
        return

    print(f"Found device: {v4l2['device']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("labeled_runs", f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "output.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "v4l2",
        "-framerate",
        "60",
        "-i",
        v4l2["device"],
        "-vf",
        VIDEO_FILTERS,
        "-c:v",
        CODEC,
        "-cq",
        CQ,
        "-rc",
        "vbr",
        "-preset",
        "p5",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        "-y",
        "-loglevel",
        "error",
        output_path,
    ]

    key_state = set()
    lock = threading.Lock()

    def on_press(k):
        with lock:
            key_state.add(key_to_name(k))

    def on_release(k):
        with lock:
            key_state.discard(key_to_name(k))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, start_new_session=True, stdin=subprocess.DEVNULL
    )

    keys_list = []
    start_time = time.time()
    frame_count = 0

    print("Recording.  Press Ctrl+C to stop.\n")

    next_tick = time.monotonic()

    try:
        while True:
            with lock:
                keys_list.append(sorted(key_state))
            frame_count += 1
            next_tick += INTERVAL
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass

    listener.stop()
    ffmpeg_proc.send_signal(signal.SIGINT)

    try:
        ffmpeg_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ffmpeg_proc.kill()
        ffmpeg_proc.wait()

    elapsed = time.time() - start_time

    # Save metadata
    with open(os.path.join(out_dir, "keys.json"), "w") as f:
        json.dump(keys_list, f)

    meta = {
        "device": v4l2["device"],
        "capture_method": "v4l2",
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "frame_count": frame_count,
        "duration_seconds": round(elapsed, 1),
        "target_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "fps": FPS,
        "codec": CODEC,
        "cq": CQ,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rate = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nSaved {frame_count} keyframes ({rate:.1f}/s) in {out_dir}")
    print(f"Duration: {elapsed:.1f}s")

    # Validate output
    print("Validating output video...")
    if validate_video(output_path):
        print("  OK - video contains real content")
    else:
        print("  WARNING - video may be black. Check the recording.")


if __name__ == "__main__":
    main()
