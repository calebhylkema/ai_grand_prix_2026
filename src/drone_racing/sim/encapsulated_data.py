"""Parsers for the sim's custom MAVLink ENCAPSULATED_DATA payloads.

The sim repurposes MAVLink's ENCAPSULATED_DATA and DATA_TRANSMISSION_HANDSHAKE
messages to broadcast race-specific data outside the standard message set.

Two payload types share the channel, discriminated by the first byte
(``data_type``):

- ``data_type == 1`` — RaceStatus (single packet)
- ``data_type == 2`` — TrackInfo chunk (chunked across multiple packets,
  preceded by a DATA_TRANSMISSION_HANDSHAKE that announces transfer id +
  chunk count)

Wire formats (from PyAIPilotExample/mavlink_rx.py)
--------------------------------------------------
RaceStatus payload (37 bytes, little-endian)::

    <B  data_type                 uint8    always 1
    <Q  sim_boot_time_ms          uint64   ms since sim boot
    <q  race_start_boot_time_ms   int64    sim time at race start (negative if not started)
    <q  race_finish_time_ns       int64    sim time at race finish (negative if ongoing)
    <I  active_gate_index         uint32   index of current target gate
    <q  last_gate_race_time       int64    race time when last gate was passed

TrackInfo chunk payload header (3 bytes)::

    <B  data_type   uint8    always 2
    <H  transfer_id uint16   identifies which TRACK transfer this belongs to

Followed by the per-chunk slice of the assembled TRACK_INFO payload.

Assembled TRACK_INFO payload::

    <H num_gates           uint16

    Then num_gates entries, each 38 bytes::
        <H gate_id          uint16
        <f position_ned_x   float32
        <f position_ned_y   float32
        <f position_ned_z   float32
        <f orientation_w    float32  (scalar-first)
        <f orientation_x    float32
        <f orientation_y    float32
        <f orientation_z    float32
        <f width            float32  (m)
        <f height           float32  (m)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------
RACE_STATUS_MSG_ID: Final[int] = 1
TRACK_INFO_MSG_ID: Final[int] = 2

_RACE_STATUS_FORMAT: Final[str] = "<BQqqIq"
_RACE_STATUS_SIZE: Final[int] = struct.calcsize(_RACE_STATUS_FORMAT)
_TRACK_HEADER_FORMAT: Final[str] = "<BH"
_TRACK_HEADER_SIZE: Final[int] = struct.calcsize(_TRACK_HEADER_FORMAT)
_GATE_ENTRY_FORMAT: Final[str] = "<Hfffffffff"
_GATE_ENTRY_SIZE: Final[int] = struct.calcsize(_GATE_ENTRY_FORMAT)


# ---------------------------------------------------------------------------
# Parsed data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RaceStatus:
    """Decoded race-status payload.

    Times follow the sim's conventions:
    ``race_start_boot_time_ms < 0`` ⇒ race not started.
    ``race_finish_time_ns < 0``     ⇒ race ongoing.
    """

    sim_boot_time_ms: int
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int
    last_gate_race_time: int

    @property
    def race_started(self) -> bool:
        """Race has been *scheduled* — countdown is running or race is live.

        ``race_start_boot_time_ms`` is set the moment the user clicks Race
        (countdown begins), not when the green-light moment arrives.  Use
        :attr:`is_live` to gate control commands.
        """
        return self.race_start_boot_time_ms >= 0

    @property
    def race_finished(self) -> bool:
        return self.race_finish_time_ns >= 0

    @property
    def is_live(self) -> bool:
        """``True`` only when the race countdown has completed and the green
        light is on.

        The sim DQs us if we apply throttle before this moment — see the
        2026-06-01 smoke test where ``race_started`` (countdown-running) was
        confused with the race-is-live moment and the drone moved during
        the countdown.
        """
        if self.race_start_boot_time_ms < 0:
            return False  # nothing scheduled
        if self.race_finish_time_ns >= 0:
            return False  # already over
        return self.sim_boot_time_ms >= self.race_start_boot_time_ms


@dataclass(frozen=True, slots=True)
class RawGateEntry:
    """A single gate entry parsed from TRACK_INFO, in sim's NED frame.

    Orientation is scalar-first ``(w, x, y, z)`` as delivered on the wire;
    convert to scalar-last and to the internal Z-up frame at the adapter
    boundary, not here.
    """

    gate_id: int
    position_ned: tuple[float, float, float]
    orientation_ned_wxyz: tuple[float, float, float, float]
    width_m: float
    height_m: float


# ---------------------------------------------------------------------------
# RaceStatus parser
# ---------------------------------------------------------------------------
def parse_race_status(payload: bytes) -> RaceStatus | None:
    """Decode a RACE_STATUS payload.

    Parameters
    ----------
    payload : bytes
        Full ENCAPSULATED_DATA payload (including the leading ``data_type``
        discriminator byte).

    Returns
    -------
    RaceStatus | None
        Parsed status, or ``None`` if the payload is malformed or has the
        wrong discriminator.
    """
    if len(payload) < _RACE_STATUS_SIZE:
        return None
    try:
        (
            data_type,
            sim_boot_time_ms,
            race_start_boot_time_ms,
            race_finish_time_ns,
            active_gate_index,
            last_gate_race_time,
        ) = struct.unpack_from(_RACE_STATUS_FORMAT, payload)
    except struct.error:
        return None
    if data_type != RACE_STATUS_MSG_ID:
        return None
    return RaceStatus(
        sim_boot_time_ms=int(sim_boot_time_ms),
        race_start_boot_time_ms=int(race_start_boot_time_ms),
        race_finish_time_ns=int(race_finish_time_ns),
        active_gate_index=int(active_gate_index),
        last_gate_race_time=int(last_gate_race_time),
    )


# ---------------------------------------------------------------------------
# Assembled TRACK_INFO parser
# ---------------------------------------------------------------------------
def parse_track_info(payload: bytes) -> list[RawGateEntry]:
    """Decode an assembled TRACK_INFO payload into raw gate entries.

    Parameters
    ----------
    payload : bytes
        Concatenated chunk payloads (after each chunk's 3-byte header has
        been stripped by :class:`TrackInfoAssembler`).

    Returns
    -------
    list[RawGateEntry]
        Gates in transmission order.  Empty if the payload is malformed.
    """
    if len(payload) < 2:
        return []
    try:
        (num_gates,) = struct.unpack_from("<H", payload)
    except struct.error:
        return []
    expected_size = 2 + num_gates * _GATE_ENTRY_SIZE
    if len(payload) < expected_size:
        logger.warning(
            "TRACK_INFO truncated: expected %d bytes, got %d", expected_size, len(payload)
        )
        return []

    gates: list[RawGateEntry] = []
    offset = 2
    for _ in range(num_gates):
        (gate_id, px, py, pz, qw, qx, qy, qz, width, height) = struct.unpack_from(
            _GATE_ENTRY_FORMAT, payload, offset
        )
        gates.append(
            RawGateEntry(
                gate_id=int(gate_id),
                position_ned=(float(px), float(py), float(pz)),
                orientation_ned_wxyz=(float(qw), float(qx), float(qy), float(qz)),
                width_m=float(width),
                height_m=float(height),
            )
        )
        offset += _GATE_ENTRY_SIZE
    return gates


# ---------------------------------------------------------------------------
# TRACK_INFO chunk assembler
# ---------------------------------------------------------------------------
@dataclass
class _TrackTransfer:
    """In-progress TRACK_INFO chunked transfer."""

    expected_chunks: int
    chunks: dict[int, bytes] = field(default_factory=dict)


class TrackInfoAssembler:
    """Reassemble a TRACK_INFO payload across multiple ENCAPSULATED_DATA packets.

    Usage::

        assembler = TrackInfoAssembler()
        assembler.handshake(transfer_id, packet_count)  # from DATA_TRANSMISSION_HANDSHAKE
        for chunk in encapsulated_data_packets:
            payload = bytes(chunk.data)
            seqnr = chunk.seqnr
            gates = assembler.add_chunk(seqnr, payload)
            if gates is not None:
                # Transfer complete — `gates` is the full list.
                ...
    """

    def __init__(self) -> None:
        self._transfers: dict[int, _TrackTransfer] = {}

    def handshake(self, transfer_id: int, expected_chunks: int) -> None:
        """Register an upcoming transfer announced by DATA_TRANSMISSION_HANDSHAKE."""
        self._transfers[int(transfer_id)] = _TrackTransfer(expected_chunks=int(expected_chunks))

    def add_chunk(self, seqnr: int, payload: bytes) -> list[RawGateEntry] | None:
        """Add one ENCAPSULATED_DATA chunk; return parsed gates if complete.

        Parameters
        ----------
        seqnr : int
            ``msg.seqnr`` of the ENCAPSULATED_DATA packet.
        payload : bytes
            Full ENCAPSULATED_DATA ``msg.data`` bytes.  Must start with the
            3-byte ``<BH`` chunk header (``data_type``, ``transfer_id``).

        Returns
        -------
        list[RawGateEntry] | None
            Parsed gates once the final chunk arrives; ``None`` while the
            transfer is still in progress, the chunk is malformed, or no
            handshake has been seen for this transfer id.
        """
        if len(payload) < _TRACK_HEADER_SIZE:
            return None
        try:
            data_type, transfer_id = struct.unpack_from(_TRACK_HEADER_FORMAT, payload)
        except struct.error:
            return None
        if data_type != TRACK_INFO_MSG_ID:
            return None

        transfer = self._transfers.get(int(transfer_id))
        if transfer is None:
            # No handshake yet — silently drop.  The sim retries.
            return None

        transfer.chunks[int(seqnr)] = bytes(payload[_TRACK_HEADER_SIZE:])

        if len(transfer.chunks) < transfer.expected_chunks:
            return None

        try:
            full = b"".join(transfer.chunks[i] for i in range(transfer.expected_chunks))
        except KeyError:
            # Missing chunk — wait for retransmit; don't tear down state.
            return None

        del self._transfers[int(transfer_id)]
        return parse_track_info(full)

    def reset(self) -> None:
        """Drop all in-progress transfers."""
        self._transfers.clear()
