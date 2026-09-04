"""Measure real capture FPS across backend / format / exposure combinations.

Run:  uv run python scripts/diagnose_camera.py

Round 2. Round 1 established that DirectShow refuses MJPG on this camera and
that 720p costs roughly 1.7x the frame time of 480p - less than pixel count
alone would explain, which points at exposure time as a second limiter.

Not part of the package - a one-off developer tool.
"""

import time

import cv2

WARMUP_FRAMES = 10
MEASURE_FRAMES = 60

BACKENDS = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}

# (label, backend, width, height, fourcc, manual_exposure)
#
# manual_exposure is the CAP_PROP_EXPOSURE value in log2 seconds on Windows:
#   -5 = 1/32s (~31ms, caps you at 32 FPS)
#   -6 = 1/64s (~16ms, allows 60 FPS)
# Locking it short is what frees the sensor to run fast; the cost is a darker
# image, which is a real tradeoff for pose detection.
CONFIGS = [
    ("MSMF 640x480 default",        "MSMF",  640, 480, None,   None),
    ("MSMF 640x480 MJPG",           "MSMF",  640, 480, "MJPG", None),
    ("MSMF 1280x720 MJPG",          "MSMF", 1280, 720, "MJPG", None),
    ("DSHOW 640x480 exposure -6",   "DSHOW",  640, 480, None,   -6),
    ("MSMF 640x480 MJPG exp -6",    "MSMF",  640, 480, "MJPG", -6),
]


def measure(backend: str, width: int, height: int,
            fourcc: str | None, exposure: float | None) -> None:
    capture = cv2.VideoCapture(0, BACKENDS[backend])
    if not capture.isOpened():
        print("  could not open camera")
        return

    try:
        if fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if exposure is not None:
            # The magic constants differ per backend: DirectShow uses 0.25 to
            # mean "manual", Media Foundation uses 0. Set both and let the
            # driver ignore the one it does not understand.
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if backend == "DSHOW" else 0)
            capture.set(cv2.CAP_PROP_EXPOSURE, exposure)

        for _ in range(WARMUP_FRAMES):
            capture.read()

        start = time.perf_counter()
        received = 0
        for _ in range(MEASURE_FRAMES):
            ok, _frame = capture.read()
            if ok:
                received += 1
        elapsed = time.perf_counter() - start

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        fmt = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        exp = capture.get(cv2.CAP_PROP_EXPOSURE)

        measured = received / elapsed if elapsed > 0 else 0.0
        print(
            f"  {actual_w}x{actual_h} format={fmt or '?'} exposure={exp:g} "
            f"-> MEASURED {measured:.1f} FPS"
        )
    finally:
        capture.release()


def main() -> None:
    print("Measuring capture FPS. Keep still, keep the lighting constant.\n")
    for label, backend, width, height, fourcc, exposure in CONFIGS:
        print(f"{label}:")
        measure(backend, width, height, fourcc, exposure)
        print()

    print(
        "Interpretation:\n"
        "  A row reaching ~30 FPS -> use exactly that configuration.\n"
        "  format=MJPG under MSMF -> DirectShow was the problem, switch backend.\n"
        "  Only the exposure rows improve -> the limit was lighting all along.\n"
        "  Nothing beats ~17 FPS  -> hardware ceiling, we design around it."
    )


if __name__ == "__main__":
    main()
