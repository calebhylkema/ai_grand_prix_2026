# Diagnose a flight

**Read the trace first.** Every root cause found in VQ2 (floor sink, glow trap,
rate inversion, gate-hopping, orbit signature) was found by reading recorded
evidence, not by watching the sim. Do this before changing any code.

## 1. Trace archives

Each attempt dumps a camera frame every 0.15 s with flight state baked into the
filename: time, roll/pitch, vz, gyro-z, gate cy/area/bearing, held target,
line-aim, and ESKF-vs-integrator speed fields (`K{h}z{vz}I{h}`).

- Current attempt: trace root under `.local/datasets/vq2/fly_trace/`
- Previous attempts: archived to `prev_YYYYMMDD_HHMMSS/` subdirectories (last 6 kept)

Read the frame **sequence** into a collision — it reconstructs approach geometry
with no human watching the sim. Claude can Read the PNGs directly.

## 2. Run summary

The mission prints per-attempt: abort reason, collision log (GATE vs ENV + threat
id + impulse), deepest phase reached, kf-vs-integrator attitude sanity
(`kf_att=(r,p)/att=(r,p)`), and vz bounds (IMU vz is **diagnostic only** — never
control on it).

## 3. Offline perception lab (no sim needed)

```bash
uv run aigp-vq2 detect-lab --annotate 8 --glob "g1_*.png"
```

Runs the gate detector over captured frame corpora and writes annotated copies —
judge detection quality frame-by-frame. This is how the glow-trap and the
two cyan-line perception bugs were found. Frame corpora live under
`.local/datasets/vq2/` (`lab/`, `fly_collisions/`, `recon_frames/`).

## 4. Collision frames

Every collision dumps the frame at impact. GATE contacts (id 1001) near a gate
plane usually mean threading precision; ENV contacts (id 1002, threat 1–2) mean
floor clutter / structures — check what filled the frame in the preceding second.

## Known signatures (match before investigating fresh)

| Signature | Meaning |
|---|---|
| Gate climbs out the frame top while floor fills the view | vertical sink — closed-loop visual height not engaged |
| PnP z jumping 27→21→17 m within 0.3 s | gate-hopping — data association latched an imposter |
| Bearing drifts away at MAX_YAW with bank clamped | orbit — likely chasing the wrong object or backward drift |
| Tilt → 60–97° at the end | the RESULT of a collision, not the cause — look earlier |
| "Brake" pitching hard while sinking | trusting a stale/fictional velocity estimate |
