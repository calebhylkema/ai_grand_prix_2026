# VQ2 deployment data

Version-controlled **data** for the VQ2 deployment. The **code** lives in the
installed package at `src/drone_racing/deployments/vq2/` — nothing here is
imported; it is read at runtime by name.

## Contents

```
configs/
  training.toml        development runs (sim in Training mode; trace recording on)
  qualification.toml   competitive runs (sim in Qualification mode; trace off —
                       runs are code-audited, keep the stack clean)
course_maps/           small, reviewed course maps mined from traces
```

## Rules

- The two configs are **complete and separate** — no layering or inheritance.
  What you read is what flies.
- Angles are stored in **degrees** in TOML (`*_deg` keys) and converted to
  radians exactly once, by the loader
  (`drone_racing.deployments.vq2.config.load`).
- Numerical flight values (gains, clamps, thresholds) are flight-validated;
  treat edits like control-code edits (see `docs/reference/safety.md`).
- Course maps here are the reviewed, version-controlled artifacts. Raw map
  observations and trace archives belong in `.local/datasets/vq2/` (gitignored).

## How configs are found

`config.load("training")` resolves against the **workspace root** (located via
`AIGP_WORKSPACE`, else an upward search for `pyproject.toml`/`.local`), then
`deployments/vq2/configs/<name>.toml`. Configs are deliberately **not**
packaged into the wheel — they are repo-bound so qualification runs audit as a
single reviewed tree. An explicit file path also works:
`aigp-vq2 --config path/to/custom.toml`.
