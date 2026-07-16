# Coordinate frames

## VQ2 simulator frames (VADR-TS-003 §3.7–3.8)

- **World:** NED; X north/local forward, Y east/local right, Z down.
- **Body:** FRD; X forward, Y right, Z down, origin at the center of mass.
- **IMU:** aligned with body FRD.
- **Camera:** OpenCV X-right, Y-down, Z-forward, sharing the body origin.

## Camera extrinsic — resolved

The camera is tilted **20° up**, matching VADR-TS-003. The earlier down-tilt reading
was confounded by the drone's roughly 18° nose-down rest/launch attitude. Projection
and coupled-renderer tests guard the corrected sign.

Camera intrinsics are pinhole/no distortion at 640×360: `fx=fy=320`, `cx=320`,
`cy=180`. Predicting gate visibility requires both body attitude and the camera
extrinsic; camera tilt alone does not establish that a gate is outside the image.

## Course-map frame

Until an absolute world pose is observable, VQ2 course maps use
`gate0_local_ned`: origin at gate 0, X along the verified gate-0 exit heading, Y
right, and Z down. This is a yaw-anchored local NED frame, not a claim that the
simulator's absolute north/east origin is known.

## Internal conventions

- SI units internally; convert once at boundaries.
- Quaternions scalar-last `[x, y, z, w]`.
- Gate map centers and planning state are NED.
- Body-rate commands use FRD signs before the calibrated command boundary.
- `pose_cam[2]` is camera depth; `norm(pose_cam)` is slant range. Never label one as
  the other.
- The pre-sim legacy MPCC model is Z-up and is not compatible with VQ2 flight state.
  `VQ2DynamicsConfig` is the NED/rate-thrust planning contract.

## Vision stream wire format

UDP 5600 uses little-endian header `<IHHIIQ>`:
`frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns`, followed by
a JPEG slice. Reassembly deduplicates completed frame IDs and preserves `sim_time_ns`.

## Gate dimensions

The VQ2 inner opening is 1.5×1.5 m, outer frame 2.7×2.7 m, depth 0.26 m, and drone
chassis approximately 280×280×160 mm. Dimensions live in configuration/course-map
data; planner code must not hardcode them.
