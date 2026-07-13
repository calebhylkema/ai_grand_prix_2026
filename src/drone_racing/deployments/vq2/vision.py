"""Robust chunked-JPEG frame reassembly for the VQ2 vision UDP stream.

The camera stream (spec §4.6, UDP :5600) sends each 640×360 JPEG as a run of chunks
behind a fixed 24-byte header. The naive reassembler both receivers used deleted a
completed frame's chunk buffer WITHOUT remembering that the frame had already been
decoded, so a late or duplicate UDP chunk re-created and re-decoded the same frame —
inflating the decoded-frame count (recon saw ~25 k "frames" in 20 s) and re-emitting
stale frames. This module centralizes a dedup-aware reassembler so the live receiver
(:func:`calibration.vision_rx`) and the diagnostics recon share one correct path:
each frame id is decoded exactly once, headers are validated, incomplete assemblies
expire, and the header ``sim_time_ns`` is retained with every completed frame.
"""

from __future__ import annotations

import collections
import struct
from dataclasses import dataclass

import cv2
import numpy as np

# frame_id u32, chunk_id u16, total_chunks u16, jpeg_size u32, payload_size u32,
# sim_time_ns u64  (spec §4.6 — 24 bytes, little-endian)
HEADER_FMT = "<IHHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

_MAX_TOTAL_CHUNKS = 4096  # sanity bound on total_chunks
_COMPLETED_MEMORY = 512  # remember this many recently-decoded frame ids (dedup window)
_PENDING_EXPIRE_SPAN = 30  # drop incomplete assemblies this many frame ids behind newest
_RESET_GAP = 1000  # a frame id this far below the max ⇒ stream restart (sim reset)
_NOMINAL_DT_NS = int(1e9 / 30)  # 30 Hz camera (spec §4.6)


@dataclass
class CompletedFrame:
    """One fully-reassembled, decoded frame (emitted exactly once per frame id)."""

    frame_id: int
    sim_time_ns: int
    image: np.ndarray


@dataclass
class ReassemblyStats:
    """Health counters for the vision stream."""

    unique_decoded: int = 0
    dup_stale_chunks: int = 0  # chunks for already-completed or too-old frame ids
    invalid_chunks: int = 0  # header/size validation failures
    incomplete_expired: int = 0  # assemblies dropped before completing
    decode_failures: int = 0  # jpeg-size mismatch or imdecode returned None
    frame_id_gaps: int = 0  # completed frame id skipped >1 from the previous
    ts_gaps: int = 0  # completed-frame sim_time gap >1.5x the nominal 30 Hz period
    resets: int = 0  # detected stream restarts
    first_ts: int | None = None
    last_ts: int | None = None

    def fps(self) -> float:
        """Unique decoded FPS from the sim-timestamp span (0 if <2 frames)."""
        if self.first_ts is None or self.last_ts is None or self.last_ts <= self.first_ts:
            return 0.0
        return (self.unique_decoded - 1) / ((self.last_ts - self.first_ts) * 1e-9)


class FrameReassembler:
    """Reassembles chunked JPEG frames, decoding each frame id exactly once.

    Feed raw UDP packets to :meth:`push`; it returns a :class:`CompletedFrame` only on
    the first completion of a frame id and ``None`` otherwise (duplicate/late/stale
    chunks, incomplete frames, or validation failures). ``stats`` accumulates health.
    """

    def __init__(self) -> None:
        self._pending: dict[int, dict] = {}
        self._completed: collections.deque[int] = collections.deque(maxlen=_COMPLETED_MEMORY)
        self._completed_set: set[int] = set()
        self._max_frame_id = -1
        self._last_completed_id: int | None = None
        self._last_ts: int | None = None
        self.stats = ReassemblyStats()

    def push(self, packet: bytes) -> CompletedFrame | None:
        """Feed one UDP packet; return a CompletedFrame only when a NEW frame completes."""
        if len(packet) < HEADER_SIZE:
            self.stats.invalid_chunks += 1
            return None
        frame_id, chunk_id, total, jpeg_size, payload_size, sim_ts = struct.unpack_from(
            HEADER_FMT, packet
        )
        payload = packet[HEADER_SIZE:]
        if (
            not (0 < total <= _MAX_TOTAL_CHUNKS)
            or chunk_id >= total
            or payload_size != len(payload)
        ):
            self.stats.invalid_chunks += 1
            return None

        # stream restart: frame ids jumped far backwards → reinitialize (keep stat totals)
        if self._max_frame_id >= 0 and frame_id < self._max_frame_id - _RESET_GAP:
            self._pending.clear()
            self._completed.clear()
            self._completed_set.clear()
            self._max_frame_id = -1
            self._last_completed_id = None
            self._last_ts = None
            self.stats.resets += 1

        if frame_id in self._completed_set:
            self.stats.dup_stale_chunks += 1  # a late chunk for an already-decoded frame
            return None
        if frame_id < self._max_frame_id - _PENDING_EXPIRE_SPAN:
            self.stats.dup_stale_chunks += 1  # a chunk for a frame whose assembly expired
            return None

        if frame_id > self._max_frame_id:
            self._max_frame_id = frame_id
            self._expire()

        entry = self._pending.get(frame_id)
        if entry is None:
            entry = {"chunks": {}, "total": total, "jpeg": jpeg_size}
            self._pending[frame_id] = entry
        entry["chunks"][chunk_id] = payload
        if len(entry["chunks"]) < entry["total"] or not all(
            i in entry["chunks"] for i in range(entry["total"])
        ):
            return None

        # complete: mark decoded FIRST (so any further chunks are deduped), then decode
        del self._pending[frame_id]
        self._mark_completed(frame_id)
        buf = b"".join(entry["chunks"][i] for i in range(entry["total"]))
        if entry["jpeg"] and len(buf) != entry["jpeg"]:
            self.stats.decode_failures += 1
            return None
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.stats.decode_failures += 1
            return None
        self._record_completion(frame_id, int(sim_ts))
        return CompletedFrame(frame_id, int(sim_ts), img)

    def _mark_completed(self, frame_id: int) -> None:
        self._completed.append(frame_id)  # bounded deque auto-evicts the oldest
        self._completed_set = set(self._completed)

    def _expire(self) -> None:
        cutoff = self._max_frame_id - _PENDING_EXPIRE_SPAN
        for fid in [f for f in self._pending if f < cutoff]:
            del self._pending[fid]
            self.stats.incomplete_expired += 1

    def _record_completion(self, frame_id: int, sim_ts: int) -> None:
        s = self.stats
        s.unique_decoded += 1
        if self._last_completed_id is not None and frame_id > self._last_completed_id + 1:
            s.frame_id_gaps += 1
        self._last_completed_id = frame_id
        if s.first_ts is None:
            s.first_ts = sim_ts
        if self._last_ts is not None and sim_ts - self._last_ts > 1.5 * _NOMINAL_DT_NS:
            s.ts_gaps += 1
        self._last_ts = sim_ts
        s.last_ts = sim_ts
