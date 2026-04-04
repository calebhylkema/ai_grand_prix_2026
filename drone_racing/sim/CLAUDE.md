# Simulator Adapter Module

## Purpose

Thin adapter layer between our autonomy stack and the DCL competition simulator API. This module isolates all simulator-specific code so the core pipeline remains simulator-agnostic.

## Design Pattern: Adapter

```python
class SimAdapter(Protocol):
    def get_frame(self) -> tuple[np.ndarray, int]:  # (image, timestamp_ns)
    def get_telemetry(self) -> TelemetryData:        # position, velocity, orientation
    def send_command(self, cmd: DroneCommand) -> None:
    def is_running(self) -> bool:
```

All other modules depend on this Protocol, NOT on the DCL API directly. When the competition releases their API (expected May 2026), implement `DCLAdapter(SimAdapter)` here.

## Pre-Sim Development

Until the real simulator is available, use `MockAdapter` for testing:
- Replays recorded image sequences from `tests/fixtures/`
- Simulates IMU data with configurable noise
- Accepts commands and simulates basic physics response

## Interface Specs

Competition interface specs due "second half of March 2026." When released:
1. Update this CLAUDE.md with actual API details
2. Implement `DCLAdapter` matching the Protocol
3. Run full integration test suite against real sim
