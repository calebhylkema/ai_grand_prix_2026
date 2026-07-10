"""Vision stream receiver for the AI Grand Prix simulator.

The sim delivers the FPV camera feed as chunked JPEG over a UDP socket
separate from the MAVLink port.  Each frame is split into one or more
packets sharing a frame id; we reassemble in-memory and decode to BGR.

Wire format (per packet, from PyAIPilotExample/vision_rx.py)
-----------------------------------------------------------
Header (24 bytes, little-endian)::

    <I  frame_id      uint32   identifier for this frame
    <H  chunk_id      uint16   sequence index within the frame
    <H  total_chunks  uint16   number of chunks for this frame
    <I  jpeg_size     uint32   full JPEG byte size
    <I  payload_size  uint32   payload byte size in this packet
    <Q  sim_time_ns   uint64   server-side capture timestamp (ns)

Followed by ``payload_size`` bytes of raw JPEG payload.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_HEADER_FORMAT: Final[str] = "<IHHIIQ"
_HEADER_SIZE: Final[int] = struct.calcsize(_HEADER_FORMAT)


@dataclass
class _PartialFrame:
    """Per-frame reassembly buffer."""

    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    received_at_ns: int
    chunks: dict[int, bytes] = field(default_factory=dict)


class VisionReceiver:
    """UDP receiver for the sim's chunked JPEG vision stream.

    Spins a background thread that reassembles incoming chunks into full
    JPEG frames, decodes them with OpenCV, and exposes the most recent
    decoded frame via :attr:`latest_frame`.

    Parameters
    ----------
    bind_ip : str
        Local IP to bind (use ``"0.0.0.0"`` for any interface).
    port : int
        Local UDP port to listen on.
    stale_frame_timeout_s : float
        Drop in-flight partial frames older than this (seconds).
    """

    def __init__(
        self,
        bind_ip: str = "0.0.0.0",
        port: int = 5600,
        stale_frame_timeout_s: float = 1.0,
    ) -> None:
        self._bind_ip = bind_ip
        self._port = port
        self._stale_ns = int(stale_frame_timeout_s * 1e9)

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_timestamp_ns: int = 0
        self._latest_frame_id: int = -1

        self._frames_received: int = 0
        self._frames_dropped: int = 0
        self._partial_frames: dict[int, _PartialFrame] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bind the socket and launch the receive thread."""
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.bind((self._bind_ip, self._port))
        sock.settimeout(0.5)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info(
            "VisionReceiver listening on %s:%d (recv buffer 1MB)",
            self._bind_ip,
            self._port,
        )

    def stop(self) -> None:
        """Signal the receive thread to exit and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------
    @property
    def latest_frame(self) -> np.ndarray | None:
        """Most recent decoded frame as BGR ndarray (or None if none received)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def latest_frame_timestamp_ns(self) -> int:
        """Server-side capture timestamp of :attr:`latest_frame` (ns)."""
        with self._lock:
            return self._latest_frame_timestamp_ns

    @property
    def latest_frame_id(self) -> int:
        """Frame id of :attr:`latest_frame` (-1 if no frame received)."""
        with self._lock:
            return self._latest_frame_id

    @property
    def frames_received(self) -> int:
        """Total number of complete frames decoded since startup."""
        return self._frames_received

    @property
    def frames_dropped(self) -> int:
        """Total number of frames dropped due to missing chunks or decode failure."""
        return self._frames_dropped

    # ------------------------------------------------------------------
    # Receive thread
    # ------------------------------------------------------------------
    def _recv_loop(self) -> None:
        """Receive packets, reassemble JPEGs, decode."""
        assert self._sock is not None
        last_gc_ns = time.monotonic_ns()
        while self._running:
            try:
                packet, _addr = self._sock.recvfrom(65536)
            except TimeoutError:
                packet = None
            except OSError:
                if self._running:
                    logger.exception("VisionReceiver socket error")
                return

            if packet is not None:
                self._handle_packet(packet)

            # Periodically evict stale partial frames so memory stays bounded
            # even if no new packets are arriving.
            now_ns = time.monotonic_ns()
            if now_ns - last_gc_ns > 200_000_000:  # 200 ms
                self._evict_stale(now_ns)
                last_gc_ns = now_ns

    def _handle_packet(self, packet: bytes) -> None:
        if len(packet) < _HEADER_SIZE:
            return
        header = packet[:_HEADER_SIZE]
        payload = packet[_HEADER_SIZE:]
        try:
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(
                _HEADER_FORMAT, header
            )
        except struct.error:
            return

        if total_chunks == 0 or payload_size == 0:
            return
        if len(payload) < payload_size:
            return

        partial = self._partial_frames.get(frame_id)
        if partial is None:
            partial = _PartialFrame(
                total_chunks=total_chunks,
                jpeg_size=jpeg_size,
                sim_time_ns=sim_time_ns,
                received_at_ns=time.monotonic_ns(),
            )
            self._partial_frames[frame_id] = partial

        partial.chunks[chunk_id] = bytes(payload[:payload_size])

        if len(partial.chunks) != partial.total_chunks:
            return

        # All chunks present — reassemble and decode.
        self._partial_frames.pop(frame_id, None)
        try:
            jpeg = b"".join(partial.chunks[i] for i in range(partial.total_chunks))
        except KeyError:
            self._frames_dropped += 1
            return
        if partial.jpeg_size and len(jpeg) != partial.jpeg_size:
            logger.debug(
                "vision frame %d size mismatch: expected %d, got %d",
                frame_id,
                partial.jpeg_size,
                len(jpeg),
            )

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self._frames_dropped += 1
            logger.debug("vision frame %d failed to decode", frame_id)
            return

        with self._lock:
            self._latest_frame = img
            self._latest_frame_timestamp_ns = int(partial.sim_time_ns)
            self._latest_frame_id = int(frame_id)
        self._frames_received += 1

    def _evict_stale(self, now_ns: int) -> None:
        """Drop in-flight partial frames older than the stale threshold."""
        stale_ids = [
            fid
            for fid, partial in self._partial_frames.items()
            if now_ns - partial.received_at_ns > self._stale_ns
        ]
        for fid in stale_ids:
            del self._partial_frames[fid]
            self._frames_dropped += 1
