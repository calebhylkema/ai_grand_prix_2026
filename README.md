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

# Launch simulation (once simulator is available)
uv run python scripts/run_sim.py --config config/default.toml
```

## Architecture

```
Camera Frame → Gate Detection → PnP Pose → ESKF Fusion → MPCC Planner → Controller → Motor Commands
```

See `docs/PLAN.md` for the full development roadmap and `docs/RESEARCH.md` for technical research.

## Claude Code

This project is optimized for development with Claude Code. See `CLAUDE.md` for project context and `.claude/` for MCP configs, slash commands, and hooks.

## License

MIT
