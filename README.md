# personal-mcp

Personal concierge MCP server — productivity tools and FastMCP transforms for SLM orchestrator RL training.

## What this is

A [FastMCP](https://gofastmcp.com) server exposing 22 tools that an LLM/SLM orchestrator can call: calendar, to-do, notes, web search, calculator, scratchpad, task queue, and sandboxed code execution. The server is designed to be:

- **Safe by default** — all state is in-memory and synthetic; no real user data is touched.
- **RL-friendly** — tools return structured, deterministic outputs; destructive actions require explicit confirmation; errors are returned as `{"error": "..."}` dicts (not exceptions) so the agent learns to handle malformed input.
- **Transform-ready** — three FastMCP transforms are wired in by default (ToolTransform for tags, BM25SearchTransform for on-demand discovery, optional CodeMode for the "write Python to orchestrate tools" pattern).

## Install

```bash
pip install -e ".[dev]"
```

This installs `fastmcp[code-mode]` and `ddgs`, plus dev tools (pytest, ruff, bandit).

## Run

```bash
# stdio (for Claude Desktop / MCP Inspector)
fastmcp run server.py

# Streamable HTTP (for ART, DeepAgents, remote clients)
fastmcp run server.py --transport streamable-http --port 8000
# or just:
python server.py
```

## Two runtime modes

Toggle with the `CODE_MODE_ENABLED` env var:

- **default** (`CODE_MODE_ENABLED=0`): clients see 3 tools — `search_tools`, `call_tool`, `reset_state` — and discover the rest on demand via BM25.
- **code mode** (`CODE_MODE_ENABLED=1`): clients see 4 meta-tools — `tags`, `search`, `get_schema`, `execute` — and write Python that chains `call_tool()` invocations inside a sandboxed Monty interpreter.

## Tools

| Category | Tool | Purpose |
|----------|------|---------|
| State | `reset_state` | Clear all server state (call between training episodes) |
| Calendar | `create_event`, `list_events`, `find_free_slots`, `reschedule_event`, `cancel_event` | Calendar management |
| To-do | `create_task`, `list_tasks`, `complete_task` | Simple to-do tasks |
| Notes | `create_note`, `search_notes` | Short notes with tag search |
| Search | `search`, `search_and_extract` | DuckDuckGo web + news search |
| Math | `calculate` | Safe AST-sandboxed arithmetic |
| Memory | `write_scratchpad`, `read_scratchpad`, `clear_scratchpad` | Working memory for multi-step plans |
| Planning | `create_task_item`, `list_task_items`, `update_task_item`, `delete_task_item` | Explicit task queue for plan decomposition |
| Code | `run_python` | Sandboxed Python subprocess (5s default timeout) |

## Test

```bash
pytest test_app.py -v
ruff check server.py test_app.py
bandit -r server.py
```
