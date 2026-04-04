# GSD (Get Shit Done) Setup Notes

GSD is a meta-prompting and spec-driven development system for Claude Code.
It adds structured phases (discuss → plan → execute → verify) to your workflow.

## Installation

```bash
# Install globally (recommended)
npx get-shit-done-cc --claude --global

# Or install per-project
npx get-shit-done-cc --claude
```

## What GSD Adds

- `.planning/` directory with PROJECT.md, ROADMAP.md, PLANS.md
- Custom slash commands for structured development phases
- `gsd-memory` MCP server for persistent project state
- Agent templates for spec-driven coding

## Integration with This Project

GSD works alongside our existing CLAUDE.md and slash commands.
The `.planning/` directory is GSD's workspace — our `docs/PLAN.md` is the
human-readable plan, while GSD's PLANS.md tracks machine-actionable tasks.

## Known Issues

- Sub-agents in GSD cannot currently inherit MCP tools from parent (Claude Code bug #13898)
- `gsd-memory` MCP may need to be configured in `.claude/mcp.json` rather than global settings
- If GSD agents can't access Context7, add Context7 MCP to project-level config

## Resources

- GitHub: https://github.com/gsd-build/get-shit-done
- Medium review: https://medium.com/@joe.njenga/i-tested-gsd-claude-code-meta-prompting-that-ships-faster
