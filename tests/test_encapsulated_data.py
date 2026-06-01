"""Tests for the sim's ENCAPSULATED_DATA payload parsers."""

from __future__ import annotations

import struct

import pytest

from drone_racing.sim.encapsulated_data import (
    RACE_STATUS_MSG_ID,
    TRACK_INFO_MSG_ID,
    RaceStatus,
    TrackInfoAssembler,
    parse_race_status,
    parse_track_info,
)


def _encode_race_status(
    *,
    sim_boot_time_ms: int = 1_000,
    race_start_boot_time_ms: int = -1,
    race_finish_time_ns: int = -1,
    active_gate_index: int = 0,
    last_gate_race_time: int = 0,
) -> bytes:
    return struct.pack(
        "<BQqqIq",
        RACE_STATUS_MSG_ID,
        sim_boot_time_ms,
        race_start_boot_time_ms,
        race_finish_time_ns,
        active_gate_index,
        last_gate_race_time,
    )


def _encode_track_payload(gates: list[dict]) -> bytes:
    buf = struct.pack("<H", len(gates))
    for g in gates:
        buf += struct.pack(
            "<Hfffffffff",
            g["gate_id"],
            g["px"],
            g["py"],
            g["pz"],
            g["qw"],
            g["qx"],
            g["qy"],
            g["qz"],
            g["width"],
            g["height"],
        )
    return buf


def _chunk_track_payload(payload: bytes, transfer_id: int, num_chunks: int) -> list[bytes]:
    chunk_size = (len(payload) + num_chunks - 1) // num_chunks
    chunks: list[bytes] = []
    for i in range(num_chunks):
        slice_ = payload[i * chunk_size : (i + 1) * chunk_size]
        chunks.append(struct.pack("<BH", TRACK_INFO_MSG_ID, transfer_id) + slice_)
    return chunks


# ---------------------------------------------------------------------------
# parse_race_status
# ---------------------------------------------------------------------------
class TestParseRaceStatus:
    def test_round_trip(self) -> None:
        payload = _encode_race_status(
            sim_boot_time_ms=12345,
            race_start_boot_time_ms=5000,
            race_finish_time_ns=-1,
            active_gate_index=3,
            last_gate_race_time=2_500_000_000,
        )
        rs = parse_race_status(payload)
        assert rs == RaceStatus(
            sim_boot_time_ms=12345,
            race_start_boot_time_ms=5000,
            race_finish_time_ns=-1,
            active_gate_index=3,
            last_gate_race_time=2_500_000_000,
        )

    def test_race_started_flag(self) -> None:
        rs = parse_race_status(_encode_race_status(race_start_boot_time_ms=0))
        assert rs is not None
        assert rs.race_started is True

    def test_race_not_started_flag(self) -> None:
        rs = parse_race_status(_encode_race_status(race_start_boot_time_ms=-1))
        assert rs is not None
        assert rs.race_started is False

    def test_race_finished_flag(self) -> None:
        rs = parse_race_status(_encode_race_status(race_finish_time_ns=1_000))
        assert rs is not None
        assert rs.race_finished is True

    def test_race_ongoing_flag(self) -> None:
        rs = parse_race_status(_encode_race_status(race_finish_time_ns=-1))
        assert rs is not None
        assert rs.race_finished is False

    def test_is_live_false_when_nothing_scheduled(self) -> None:
        rs = parse_race_status(_encode_race_status(race_start_boot_time_ms=-1))
        assert rs is not None
        assert rs.is_live is False

    def test_is_live_false_during_countdown(self) -> None:
        # sim_boot_time hasn't caught up to the scheduled race start yet.
        rs = parse_race_status(
            _encode_race_status(
                sim_boot_time_ms=10_000,
                race_start_boot_time_ms=13_000,  # 3s from now
                race_finish_time_ns=-1,
            )
        )
        assert rs is not None
        assert rs.race_started is True  # countdown is running
        assert rs.is_live is False  # but the green light hasn't fired

    def test_is_live_true_after_green_light(self) -> None:
        rs = parse_race_status(
            _encode_race_status(
                sim_boot_time_ms=15_000,
                race_start_boot_time_ms=13_000,  # 2s ago
                race_finish_time_ns=-1,
            )
        )
        assert rs is not None
        assert rs.is_live is True

    def test_is_live_false_after_race_finished(self) -> None:
        rs = parse_race_status(
            _encode_race_status(
                sim_boot_time_ms=20_000,
                race_start_boot_time_ms=13_000,
                race_finish_time_ns=18_000_000_000,
            )
        )
        assert rs is not None
        assert rs.is_live is False

    def test_rejects_wrong_discriminator(self) -> None:
        bad = bytes([99]) + _encode_race_status()[1:]
        assert parse_race_status(bad) is None

    def test_rejects_truncated(self) -> None:
        assert parse_race_status(b"\x01" + b"\x00" * 5) is None


# ---------------------------------------------------------------------------
# parse_track_info
# ---------------------------------------------------------------------------
class TestParseTrackInfo:
    def test_single_gate(self) -> None:
        gates = [
            {
                "gate_id": 7,
                "px": 1.0,
                "py": 2.0,
                "pz": -3.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.5,
                "height": 1.5,
            }
        ]
        parsed = parse_track_info(_encode_track_payload(gates))
        assert len(parsed) == 1
        g = parsed[0]
        assert g.gate_id == 7
        assert g.position_ned == pytest.approx((1.0, 2.0, -3.0))
        assert g.orientation_ned_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
        assert g.width_m == pytest.approx(1.5)
        assert g.height_m == pytest.approx(1.5)

    def test_multiple_gates_preserve_order(self) -> None:
        gates = [
            {
                "gate_id": i,
                "px": float(i),
                "py": 0.0,
                "pz": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.0,
                "height": 1.0,
            }
            for i in range(5)
        ]
        parsed = parse_track_info(_encode_track_payload(gates))
        assert [g.gate_id for g in parsed] == [0, 1, 2, 3, 4]
        assert [g.position_ned[0] for g in parsed] == pytest.approx([0, 1, 2, 3, 4])

    def test_empty_payload(self) -> None:
        assert parse_track_info(b"") == []

    def test_truncated_payload(self) -> None:
        # Claims 2 gates but only provides bytes for 1 → return [] safely.
        gates_one = [
            {
                "gate_id": 0,
                "px": 0.0,
                "py": 0.0,
                "pz": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.0,
                "height": 1.0,
            }
        ]
        buf = struct.pack("<H", 2) + _encode_track_payload(gates_one)[2:]
        assert parse_track_info(buf) == []


# ---------------------------------------------------------------------------
# TrackInfoAssembler
# ---------------------------------------------------------------------------
class TestTrackInfoAssembler:
    def test_single_chunk_transfer(self) -> None:
        assembler = TrackInfoAssembler()
        gates_dict = [
            {
                "gate_id": 1,
                "px": 1.0,
                "py": 0.0,
                "pz": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.5,
                "height": 1.5,
            }
        ]
        payload = _encode_track_payload(gates_dict)
        chunks = _chunk_track_payload(payload, transfer_id=42, num_chunks=1)
        assembler.handshake(42, 1)

        result = assembler.add_chunk(seqnr=0, payload=chunks[0])
        assert result is not None
        assert len(result) == 1
        assert result[0].gate_id == 1

    def test_multi_chunk_transfer_ordered(self) -> None:
        assembler = TrackInfoAssembler()
        gates_dict = [
            {
                "gate_id": i,
                "px": float(i),
                "py": 0.0,
                "pz": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.5,
                "height": 1.5,
            }
            for i in range(4)
        ]
        payload = _encode_track_payload(gates_dict)
        chunks = _chunk_track_payload(payload, transfer_id=7, num_chunks=3)
        assembler.handshake(7, 3)

        # In-order delivery
        assert assembler.add_chunk(0, chunks[0]) is None
        assert assembler.add_chunk(1, chunks[1]) is None
        result = assembler.add_chunk(2, chunks[2])
        assert result is not None
        assert [g.gate_id for g in result] == [0, 1, 2, 3]

    def test_multi_chunk_transfer_out_of_order(self) -> None:
        assembler = TrackInfoAssembler()
        gates_dict = [
            {
                "gate_id": i,
                "px": float(i),
                "py": 0.0,
                "pz": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "width": 1.5,
                "height": 1.5,
            }
            for i in range(6)
        ]
        payload = _encode_track_payload(gates_dict)
        chunks = _chunk_track_payload(payload, transfer_id=99, num_chunks=4)
        assembler.handshake(99, 4)

        assert assembler.add_chunk(3, chunks[3]) is None
        assert assembler.add_chunk(0, chunks[0]) is None
        assert assembler.add_chunk(2, chunks[2]) is None
        result = assembler.add_chunk(1, chunks[1])
        assert result is not None
        assert [g.gate_id for g in result] == [0, 1, 2, 3, 4, 5]

    def test_chunk_without_handshake_dropped(self) -> None:
        assembler = TrackInfoAssembler()
        payload = struct.pack("<BH", TRACK_INFO_MSG_ID, 123) + b"\x00" * 10
        assert assembler.add_chunk(0, payload) is None

    def test_wrong_discriminator_ignored(self) -> None:
        assembler = TrackInfoAssembler()
        assembler.handshake(1, 1)
        bad = struct.pack("<BH", 99, 1) + b"\x00" * 4
        assert assembler.add_chunk(0, bad) is None
