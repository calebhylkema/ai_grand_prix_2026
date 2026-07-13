"""Tests for the dedup-aware vision frame reassembler."""

from __future__ import annotations

import struct

import cv2
import numpy as np

from drone_racing.deployments.vq2 import vision
from drone_racing.deployments.vq2.vision import FrameReassembler


def _frame_packets(
    frame_id: int, sim_ts: int, n_chunks: int = 3, seed: int = 0
) -> tuple[list[bytes], np.ndarray]:
    """Build the UDP packets for one small JPEG frame split into ``n_chunks`` chunks."""
    img = np.random.default_rng(seed).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    data = enc.tobytes()
    step = (len(data) + n_chunks - 1) // n_chunks
    chunks = [data[i * step : (i + 1) * step] for i in range(n_chunks)]
    chunks = [c for c in chunks if c] or [data]
    n = len(chunks)
    pkts = [
        struct.pack(vision.HEADER_FMT, frame_id, cid, n, len(data), len(ch), sim_ts) + ch
        for cid, ch in enumerate(chunks)
    ]
    return pkts, img


def test_reassembler_decodes_once_and_dedups_late_chunks() -> None:
    r = FrameReassembler()
    pkts, _ = _frame_packets(10, 1000)
    outs = [r.push(p) for p in pkts]
    completed = [o for o in outs if o is not None]
    assert len(completed) == 1
    assert completed[0].frame_id == 10 and completed[0].sim_time_ns == 1000
    assert completed[0].image is not None
    # replaying every chunk again (late/duplicate) must NOT re-decode the frame
    before = r.stats.unique_decoded
    for p in pkts:
        assert r.push(p) is None
    assert r.stats.unique_decoded == before
    assert r.stats.dup_stale_chunks >= len(pkts)


def test_reassembler_rejects_invalid_headers() -> None:
    r = FrameReassembler()
    assert r.push(b"\x00\x01\x02") is None  # shorter than the header
    assert (
        r.push(struct.pack(vision.HEADER_FMT, 1, 5, 3, 100, 4, 0) + b"xxxx") is None
    )  # cid>=total
    assert (
        r.push(struct.pack(vision.HEADER_FMT, 1, 0, 3, 100, 99, 0) + b"xxxx") is None
    )  # size≠payload
    assert r.stats.invalid_chunks == 3
    assert r.stats.unique_decoded == 0


def test_reassembler_expires_incomplete_assemblies() -> None:
    r = FrameReassembler()
    partial, _ = _frame_packets(0, 0)
    r.push(partial[0])  # only chunk 0 of frame 0 → never completes
    for fid in range(1, vision._PENDING_EXPIRE_SPAN + 5):  # newer frames push it out of range
        for p in _frame_packets(fid, fid * 1000)[0]:
            r.push(p)
    assert r.stats.incomplete_expired >= 1


def test_reassembler_handles_stream_reset() -> None:
    r = FrameReassembler()
    for p in _frame_packets(5000, 5_000_000)[0]:
        r.push(p)
    assert r.stats.unique_decoded == 1
    # sim restarts: frame ids jump far backwards — must decode fresh, not treat as stale
    outs = [r.push(p) for p in _frame_packets(2, 2000)[0]]
    assert any(o is not None for o in outs)
    assert r.stats.resets >= 1


def test_reassembler_reports_fps_and_gaps() -> None:
    r = FrameReassembler()
    dt = int(1e9 / 30)
    for k in range(10):
        for p in _frame_packets(k, k * dt, seed=k)[0]:
            r.push(p)
    assert r.stats.unique_decoded == 10
    assert abs(r.stats.fps() - 30.0) < 1.0
    assert r.stats.frame_id_gaps == 0
