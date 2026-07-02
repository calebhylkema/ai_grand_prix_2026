# AI Grand Prix — Autonomous Drone Racing Stack

Autonomous FPV drone racing stack for the [Anduril AI Grand Prix 2026](https://theaigrandprix.com/).

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/YOUR_USERNAME/ai-grand-prix.git
cd ai-grand-prix
uv sync --all-extras

# Run tests
uv run pytest

# VQ2 (active): verify control primitives against the sim (Training mode, run live)
uv run python scripts/vq2/control_check.py --test rates --calib

# Mock-sim pipeline (offline development)
uv run python scripts/pre_sim/run_sim.py --config config/default.toml
```

## Architecture

```
Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → Guidance/MPCC → Rate Commands (MAVLink)
```

See `docs/ROADMAP.txt` for the development roadmap and status, `docs/RESEARCH.txt` for
technical research, `docs/CODE_MAP.md` for what lives where, and `NEXT_SESSION.md` for
the current working-session handoff.

## Claude Code

This project is optimized for development with Claude Code. See `CLAUDE.md` for project context and `.claude/` for MCP configs, slash commands, and hooks.

## License

MIT
