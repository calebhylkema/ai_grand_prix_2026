"""One-off: run gates.detect_line over fly_trace frames and annotate the results.

Validates the cyan racing-line detector offline (no sim). Writes annotated frames
to data/vq2/lab_line/ — Read them to judge quality. Delete after use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gates

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "data" / "vq2" / "fly_trace"
OUT = ROOT / "data" / "vq2" / "lab_line"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    frames = sorted(TRACE.glob("*.png"))
    print(f"{len(frames)} trace frames")
    hits = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        line = gates.detect_line(img)
        if line is None:
            print(f"{f.name[:26]:28s} no-line")
            continue
        hits += 1
        cv2.circle(img, (int(line.cx), int(line.cy)), 8, (0, 255, 255), 2)
        cv2.circle(img, (int(line.far_cx), int(line.far_cy)), 6, (255, 0, 255), 2)
        cv2.circle(img, (int(line.aim_cx), int(line.aim_cy)), 6, (0, 255, 0), 2)  # AIM
        cv2.putText(
            img,
            f"a={line.area:.0f} aim={line.aim_bearing:+.2f} far={line.far_bearing:+.2f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        mask = gates.cyan_mask(img)
        img[mask > 0] = (255, 255, 0)  # paint accepted+rejected cyan pixels for eyeballing
        cv2.imwrite(str(OUT / f.name), img)
        print(f"{f.name[:26]:28s} LINE a={line.area:6.0f} cx={line.cx:3.0f} cy={line.cy:3.0f}")
    print(f"{hits}/{len(frames)} frames with a detected line -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
