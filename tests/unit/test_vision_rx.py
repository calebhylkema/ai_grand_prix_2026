"""Tests for the vision UDP chunked-JPEG receiver."""

from __future__ import annotations

import socket
import struct
import time

import cv2
import numpy as np
import pytest

from drone_racing.sim.vision_rx import VisionReceiver

_HEADER_FORMAT = "<IHHIIQ"


def _encode_jpeg(width: int = 64, height: int = 48) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # solid green
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _packetize(jpeg: bytes, *, frame_id: int, chunk_size: int, sim_time_ns: int) -> list[bytes]:
    total = (len(jpeg) + chunk_size - 1) // chunk_size
    packets: list[bytes] = []
    for i in range(total):
        slice_ = jpeg[i * chunk_size : (i + 1) * chunk_size]
        header = struct.pack(
            _HEADER_FORMAT,
            frame_id,
            i,
            total,
            len(jpeg),
            len(slice_),
            sim_time_ns,
        )
        packets.append(header + slice_)
    return packets


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def receiver():
    port = _free_port()
    rx = VisionReceiver(bind_ip="127.0.0.1", port=port)
    rx.start()
    yield rx, port
    rx.stop()


def _send_all(packets: list[bytes], port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for pkt in packets:
            s.sendto(pkt, ("127.0.0.1", port))


def _wait_for_frame(rx: VisionReceiver, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if rx.frames_received >= 1:
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for vision frame")


class TestVisionReceiver:
    def test_single_packet_frame(self, receiver) -> None:
        rx, port = receiver
        jpeg = _encode_jpeg()
        packets = _packetize(jpeg, frame_id=1, chunk_size=len(jpeg), sim_time_ns=42)
        assert len(packets) == 1
        _send_all(packets, port)

        _wait_for_frame(rx)
        frame = rx.latest_frame
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        assert rx.latest_frame_timestamp_ns == 42
        assert rx.latest_frame_id == 1

    def test_multi_chunk_frame_in_order(self, receiver) -> None:
        rx, port = receiver
        jpeg = _encode_jpeg(width=128, height=96)
        packets = _packetize(jpeg, frame_id=5, chunk_size=300, sim_time_ns=100)
        assert len(packets) > 1
        _send_all(packets, port)

        _wait_for_frame(rx)
        frame = rx.latest_frame
        assert frame is not None
        assert frame.shape == (96, 128, 3)

    def test_multi_chunk_frame_reverse_order(self, receiver) -> None:
        rx, port = receiver
        jpeg = _encode_jpeg(width=128, height=96)
        packets = _packetize(jpeg, frame_id=9, chunk_size=200, sim_time_ns=7)
        _send_all(list(reversed(packets)), port)

        _wait_for_frame(rx)
        frame = rx.latest_frame
        assert frame is not None
        assert rx.latest_frame_id == 9

    def test_frames_dropped_on_missing_chunk(self, receiver) -> None:
        rx, port = receiver
        jpeg = _encode_jpeg(width=128, height=96)
        packets = _packetize(jpeg, frame_id=21, chunk_size=200, sim_time_ns=1)
        assert len(packets) >= 3
        # Drop the second chunk; let the partial frame age out.
        del packets[1]
        _send_all(packets, port)

        # Wait long enough for stale-frame eviction (>1s default).
        time.sleep(1.5)
        assert rx.frames_received == 0
        assert rx.frames_dropped >= 1

    def test_latest_frame_returns_copy(self, receiver) -> None:
        rx, port = receiver
        jpeg = _encode_jpeg()
        _send_all(_packetize(jpeg, frame_id=1, chunk_size=len(jpeg), sim_time_ns=1), port)
        _wait_for_frame(rx)

        a = rx.latest_frame
        b = rx.latest_frame
        assert a is not None and b is not None
        a[0, 0, 0] = 123
        assert b[0, 0, 0] != 123  # caller mutation must not leak
