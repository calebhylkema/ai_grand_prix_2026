# Coordinate frames

## VQ2 sim frames (spec VADR-TS-003 §3.7–3.8)

- **World:** NED (`MAV_FRAME_LOCAL_NED`), origin = point on ground where the drone armed.
- **Body:** FRD (X fwd, Y right, Z down), origin at center of mass. **Body → IMU = identity.**
- **Camera:** OpenCV convention (X-right, Y-down, Z-forward), shares the body origin,
  tilted 20° about pitch from the body axis.

## 🚨 Camera tilt sign — OPEN QUESTION

- The **spec says 20° UP**.
- The **code implements 20° DOWN**, based on a 2026-07-03 live-imagery reading
  (level hover → horizon at y≈64 with floor filling the frame).
- These cannot both be right; a wrong sign geometrically inverts every PnP pose.
  See [status.md](../project/status.md) for the validation plan. Update this page
  when resolved.

## Camera intrinsics (VQ2, hardcoded in the deployment)

Pinhole, no distortion. 640×360 @ 30 Hz, `[cx,cy] = [320,180]`, `[fx,fy] = [320,320]`,
VFoV 90°. With 20° down-tilt, frame row 0 is only ~9° above body-horizontal — an
elevated gate can be physically out of frame exactly when needed (a root cause of
the gate-1 blindness; see ADR-0003).

## Internal conventions

- SI units internally (meters, radians, seconds); convert at boundaries only.
- Quaternions scalar-last `[x, y, z, w]`, except scipy interfaces (scalar-first).
- Gate poses `[x, y, z, qx, qy, qz, qw]` in world frame.
- The pre-sim package pipeline used internal Z-up with NED conversion at the adapter
  boundary (`utils/transforms.py: ned_to_enu` etc.). The VQ2 deployment works in
  body/camera-relative terms and mostly avoids a world frame.

## Vision stream wire format (UDP 5600)

Little-endian, 24-byte header `<IHHIIQ` =
`frame_id(u32), chunk_id(u16), total_chunks(u16), jpeg_size(u32), payload_size(u32), sim_time_ns(u64)`,
followed by a JPEG slice. Reassemble chunks by `frame_id`.

## Gate dimensions (VQ2)

Inner opening (fly-through) **1.5 × 1.5 m**, outer frame 2.7 × 2.7 m, depth 0.26 m.
Threading tolerance ≈ ±0.75 m half-width. Drone chassis 280×280×160 mm. Read gate
dimensions from config — never hardcode.
