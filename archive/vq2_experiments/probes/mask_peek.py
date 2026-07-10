"""One-off: visualize red-mask variants + contour geometry on a few frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import detect_lab as dl
import numpy as np
import race as rc

OUT = Path(__file__).resolve().parents[2] / "data" / "vq2" / "lab"


def peek(path: Path) -> None:
    img = cv2.imread(str(path))
    m1 = rc._red_mask(img)
    m2 = dl.red_mask_v2(img)
    vis = np.zeros_like(img)
    vis[..., 2] = m1  # red channel = v1 mask
    vis[..., 1] = m2  # green channel = v2 mask (yellow where both)
    contours, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        nv = len(cv2.approxPolyDP(c, 0.04 * peri, True))
        core = m1[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]
        fill = float(np.count_nonzero(core)) / max(1, core.size)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(
            vis,
            f"a={area:.0f} v={nv} fill={fill:.2f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    out = OUT / f"mask_{path.name}"
    cv2.imwrite(str(out), vis)
    print(f"wrote {out}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        peek(Path(arg))
