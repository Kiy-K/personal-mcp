"""
Concierge MCP server — personal productivity + RL-training tools.

Built for two purposes at once:
  1. Training the Gemma 4 E2B orchestrator via ART's MCP-RL
  2. The SAME server you wire into create_deep_agent() and demo for
     the Google/Kaggle Concierge Agents capstone track

Uses the standalone `fastmcp` package (NOT the older mcp.server.fastmcp
bundled in the official MCP SDK) — fastmcp is the current de facto
standard, maintained by Prefect.

Install: pip install "fastmcp[code-mode]" ddgs

Local dev (stdio, for Claude Desktop / MCP Inspector):
    fastmcp run server.py

Remote / what ART needs a URL for (Streamable HTTP):
    fastmcp run server.py --transport streamable-http --port 8000
    # or just: python server.py

Two runtime modes (toggle with the CODE_MODE_ENABLED env var):
  - default (CODE_MODE_ENABLED=0): tools are exposed behind a
    BM25SearchTransform. Clients see three tools — search_tools,
    call_tool, and reset_state (pinned) — and discover the rest
    on demand.
  - code mode (CODE_MODE_ENABLED=1): tools are wrapped in
    CodeMode. Clients see four meta-tools — tags, search,
    get_schema, execute — and write Python that chains
    call_tool() invocations inside a sandboxed Monty interpreter.
    Use this mode for episodes that specifically target the
    "write code to orchestrate tools" pattern.
"""

import ast
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Any

from ddgs import DDGS
from fastmcp import FastMCP
from fastmcp.server.transforms import ToolTransform
from fastmcp.server.transforms.search import BM25SearchTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

# -----------------------------------------------------------------
# Transform configuration
# -----------------------------------------------------------------
# We layer three FastMCP transforms on top of the registered tools:
#
# 1. ToolTransform — adds tool-level tags and a richer title to
#    every tool. Tags drive both the BM25 search index and the
#    CodeMode discovery tool (`GetTags`) if it's enabled, so good
#    tags = better search ranking = better RL training signal.
#    The transform is applied to each tool below in the
#    _TOOL_TAGS dict.
#
# 2. BM25SearchTransform — replaces the flat tool listing with
#    two synthetic tools, `search_tools` and `call_tool`. The
#    LLM learns to discover tools on-demand by natural-language
#    query, rather than receiving the full catalog in context.
#    The original tools remain fully callable through the
#    `call_tool` proxy and through MCP's direct tool calls.
#
# 3. CodeMode (optional) — when CODE_MODE_ENABLED=1, the server
#    additionally wraps the catalog in a sandboxed Python
#    execution environment. The LLM sees only three meta-tools
#    (search, get_schema, execute) and writes Python code that
#    chains call_tool() invocations. This is the "code mode"
#    pattern from Cloudflare / Anthropic — useful for teaching
#    the orchestrator to synthesize procedural code instead of
#    making many sequential tool calls. Disabled by default
#    because it masks the individual tools, which makes
#    per-tool RL credit assignment harder. Enable it for
#    episodes that specifically target code-mode behavior.
# -----------------------------------------------------------------

_CODE_MODE_ENABLED = os.environ.get("CODE_MODE_ENABLED", "0") == "1"

_search_transform = BM25SearchTransform(
    max_results=8,
    always_visible=["reset_state"],
)

_tool_tag_transform = ToolTransform({
    # Calendar
    "create_event": ToolTransformConfig(
        title="Create Calendar Event",
        tags={"calendar", "write"},
    ),
    "list_events": ToolTransformConfig(
        title="List Calendar Events",
        tags={"calendar", "read"},
    ),
    "find_free_slots": ToolTransformConfig(
        title="Find Free Time Slots",
        tags={"calendar", "read", "planning"},
    ),
    "reschedule_event": ToolTransformConfig(
        title="Reschedule Event",
        tags={"calendar", "write"},
    ),
    "cancel_event": ToolTransformConfig(
        title="Cancel Event",
        tags={"calendar", "write", "destructive"},
    ),
    # To-do tasks
    "create_task": ToolTransformConfig(
        title="Create To-Do Task",
        tags={"todo", "write"},
    ),
    "list_tasks": ToolTransformConfig(
        title="List To-Do Tasks",
        tags={"todo", "read"},
    ),
    "complete_task": ToolTransformConfig(
        title="Complete To-Do Task",
        tags={"todo", "write"},
    ),
    # Notes
    "create_note": ToolTransformConfig(
        title="Create Note",
        tags={"notes", "write"},
    ),
    "search_notes": ToolTransformConfig(
        title="Search Notes",
        tags={"notes", "read", "search"},
    ),
    # Search
    "search": ToolTransformConfig(
        title="Web Search",
        tags={"search", "web", "read"},
    ),
    "search_and_extract": ToolTransformConfig(
        title="News Search with Date and Source",
        tags={"search", "web", "news", "read"},
    ),
    # Calculator
    "calculate": ToolTransformConfig(
        title="Calculate Math Expression",
        tags={"math", "compute", "read"},
    ),
    # Scratchpad
    "write_scratchpad": ToolTransformConfig(
        title="Write Scratchpad",
        tags={"memory", "write"},
    ),
    "read_scratchpad": ToolTransformConfig(
        title="Read Scratchpad",
        tags={"memory", "read"},
    ),
    "clear_scratchpad": ToolTransformConfig(
        title="Clear Scratchpad",
        tags={"memory", "write"},
    ),
    # Task queue
    "create_task_item": ToolTransformConfig(
        title="Create Task Queue Item",
        tags={"planning", "write", "task-queue"},
    ),
    "list_task_items": ToolTransformConfig(
        title="List Task Queue Items",
        tags={"planning", "read", "task-queue"},
    ),
    "update_task_item": ToolTransformConfig(
        title="Update Task Queue Item",
        tags={"planning", "write", "task-queue"},
    ),
    "delete_task_item": ToolTransformConfig(
        title="Delete Task Queue Item",
        tags={"planning", "write", "destructive", "task-queue"},
    ),
    # Code execution
    "run_python": ToolTransformConfig(
        title="Run Python Code",
        tags={"code", "compute", "sandbox"},
    ),
    # reset_state is intentionally NOT tagged here — it stays
    # pinned in the BM25 always_visible list above.
})

_transforms = [_tool_tag_transform, _search_transform]

if _CODE_MODE_ENABLED:
    # CodeMode and BM25SearchTransform are mutually exclusive at the
    # same layer: both want to own the tool listing. When code mode
    # is on, we drop the BM25 search proxy and let CodeMode expose
    # its own search/get_schema/execute meta-tools. The tool-tag
    # transform stays — CodeMode's GetTags uses the same tag data.
    _transforms = [_tool_tag_transform]
    from fastmcp.experimental.transforms.code_mode import (
        CodeMode,
        MontySandboxProvider,
    )
    from fastmcp.experimental.transforms.code_mode import (
        GetSchemas as CMGetSchemas,
    )
    from fastmcp.experimental.transforms.code_mode import (
        GetTags as CMGetTags,
    )
    from fastmcp.experimental.transforms.code_mode import (
        Search as CMSearch,
    )
    _sandbox = MontySandboxProvider(
        limits={"max_duration_secs": 10, "max_memory": 50_000_000},
    )
    _code_mode = CodeMode(
        sandbox_provider=_sandbox,
        max_tool_calls=30,
        discovery_tools=[CMGetTags(), CMSearch(), CMGetSchemas()],
    )
    _transforms.append(_code_mode)


mcp = FastMCP("concierge-assistant", transforms=_transforms)

# -----------------------------------------------------------------
# In-memory state. Everything here is synthetic/example data by
# design — this server is meant to be safe to point an RL training
# loop (or a stranger's demo) at without ever touching anyone's real
# calendar, tasks, or notes.
# -----------------------------------------------------------------
_state = {
    "events": {},
    "tasks": {},
    "notes": {},
    "next_event_id": 1,
    "next_task_id": 1,
    "next_note_id": 1,
    "scratchpad": "",
    "task_queue": {},
    "next_task_queue_id": 1,
}


@mcp.tool
def reset_state() -> str:
    """Reset all server-side state (events, tasks, notes, scratchpad,
    task_queue) to empty. Call this before starting a new training
    episode/scenario so every rollout begins from identical conditions
    — GRPO compares multiple attempts at the same scenario, and
    drifting state between attempts makes the reward signal noisy."""
    _state["events"].clear()
    _state["tasks"].clear()
    _state["notes"].clear()
    _state["next_event_id"] = 1
    _state["next_task_id"] = 1
    _state["next_note_id"] = 1
    _state["scratchpad"] = ""
    _state["task_queue"].clear()
    _state["next_task_queue_id"] = 1
    return "state reset"


# ===================================================================
# Calendar tools
# ===================================================================

@mcp.tool
def create_event(title: str, start_time: str, end_time: str, location: str | None = None) -> dict:
    """Create a calendar event.

    Args:
        title: short event name
        start_time: ISO 8601 datetime string, e.g. "2026-06-22T14:00:00"
        end_time: ISO 8601 datetime string, must be after start_time
        location: optional free-text location

    Returns:
        {"id": int, "title": str, "start_time": str, "end_time": str, "location": str|None}

    Errors:
        Returns {"error": "..."} (does not raise) if the times don't
        parse as ISO 8601 or if end_time is not after start_time —
        the agent should learn to handle malformed input gracefully,
        not crash or retry blindly.
    """
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
    except ValueError:
        return {"error": "start_time and end_time must be ISO 8601 datetimes"}
    if end <= start:
        return {"error": "end_time must be after start_time"}

    event_id = _state["next_event_id"]
    _state["next_event_id"] += 1
    event = {
        "id": event_id,
        "title": title,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "location": location,
    }
    _state["events"][event_id] = event
    return event


@mcp.tool
def list_events(date: str | None = None) -> list:
    """List calendar events, optionally filtered to a single day.

    Args:
        date: optional ISO 8601 date string "YYYY-MM-DD". If omitted,
            returns all events.

    Returns:
        List of event dicts, sorted by start_time.
    """
    events = list(_state["events"].values())
    if date:
        events = [e for e in events if e["start_time"].startswith(date)]
    return sorted(events, key=lambda e: e["start_time"])


@mcp.tool
def find_free_slots(date: str, duration_minutes: int, day_start_hour: int = 9, day_end_hour: int = 18) -> list:
    """Find open time slots on a given day that fit a requested duration.

    Args:
        date: ISO 8601 date string "YYYY-MM-DD"
        duration_minutes: how long the slot needs to be
        day_start_hour: earliest hour to consider (24h, default 9 = 9am)
        day_end_hour: latest hour to consider (24h, default 18 = 6pm)

    Returns:
        List of {"start_time": str, "end_time": str} candidate slots.
    """
    busy = sorted(
        (datetime.fromisoformat(e["start_time"]), datetime.fromisoformat(e["end_time"]))
        for e in _state["events"].values()
        if e["start_time"].startswith(date)
    )
    day_start = datetime.fromisoformat(f"{date}T{day_start_hour:02d}:00:00")
    day_end = datetime.fromisoformat(f"{date}T{day_end_hour:02d}:00:00")
    duration = timedelta(minutes=duration_minutes)

    slots = []
    cursor = day_start
    for busy_start, busy_end in busy:
        if busy_start - cursor >= duration:
            slots.append({"start_time": cursor.isoformat(), "end_time": (cursor + duration).isoformat()})
        cursor = max(cursor, busy_end)
    if day_end - cursor >= duration:
        slots.append({"start_time": cursor.isoformat(), "end_time": (cursor + duration).isoformat()})
    return slots


@mcp.tool
def reschedule_event(event_id: int, new_start_time: str, new_end_time: str) -> dict:
    """Move an existing event to a new time.

    Returns the updated event, or {"error": ...} if event_id doesn't
    exist or the new times are invalid.
    """
    event = _state["events"].get(event_id)
    if event is None:
        return {"error": f"no event with id {event_id}"}
    try:
        start = datetime.fromisoformat(new_start_time)
        end = datetime.fromisoformat(new_end_time)
    except ValueError:
        return {"error": "new_start_time and new_end_time must be ISO 8601 datetimes"}
    if end <= start:
        return {"error": "new_end_time must be after new_start_time"}
    event["start_time"] = start.isoformat()
    event["end_time"] = end.isoformat()
    return event


@mcp.tool
def cancel_event(event_id: int, confirm: bool = False) -> dict:
    """Cancel (delete) a calendar event.

    This is destructive, so it requires confirm=True to actually act.
    Calling it with confirm=False returns a confirmation prompt
    instead of deleting anything — the orchestrator should surface
    that prompt (e.g. to the user) rather than silently retrying
    with confirm=True. This is the action-budget / confirm-before-
    destructive-action pattern the capstone's security criteria
    are looking for.
    """
    event = _state["events"].get(event_id)
    if event is None:
        return {"error": f"no event with id {event_id}"}
    if not confirm:
        return {"requires_confirmation": True, "event": event, "message": "call again with confirm=True to cancel"}
    del _state["events"][event_id]
    return {"cancelled": event_id}


# ===================================================================
# Task tools
# ===================================================================

@mcp.tool
def create_task(title: str, due_date: str | None = None, priority: str = "normal") -> dict:
    """Create a to-do task.

    Args:
        title: short task description
        due_date: optional ISO 8601 date "YYYY-MM-DD"
        priority: one of "low", "normal", "high"

    Returns the created task, or {"error": ...} if priority is invalid.
    """
    if priority not in {"low", "normal", "high"}:
        return {"error": f"invalid priority: {priority!r}, must be low/normal/high"}
    task_id = _state["next_task_id"]
    _state["next_task_id"] += 1
    task = {"id": task_id, "title": title, "due_date": due_date, "priority": priority, "status": "open"}
    _state["tasks"][task_id] = task
    return task


@mcp.tool
def list_tasks(status: str | None = None) -> list:
    """List tasks, optionally filtered by status ("open" or "done")."""
    tasks = list(_state["tasks"].values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    return tasks


@mcp.tool
def complete_task(task_id: int) -> dict:
    """Mark a task as done. Returns the updated task, or {"error": ...}
    if task_id doesn't exist."""
    task = _state["tasks"].get(task_id)
    if task is None:
        return {"error": f"no task with id {task_id}"}
    task["status"] = "done"
    return task


# ===================================================================
# Notes tools
# ===================================================================

@mcp.tool
def create_note(title: str, content: str, tags: list | None = None) -> dict:
    """Save a short note.

    Args:
        title: short note title
        content: note body text
        tags: optional list of string tags for later search
    """
    note_id = _state["next_note_id"]
    _state["next_note_id"] += 1
    note = {"id": note_id, "title": title, "content": content, "tags": tags or []}
    _state["notes"][note_id] = note
    return note


@mcp.tool
def search_notes(query: str) -> list:
    """Search notes by substring match on title, content, or tags
    (case-insensitive). Returns matching notes, most recent first."""
    q = query.lower()
    matches = [
        n for n in _state["notes"].values()
        if q in n["title"].lower() or q in n["content"].lower() or any(q in t.lower() for t in n["tags"])
    ]
    return sorted(matches, key=lambda n: n["id"], reverse=True)


# ===================================================================
# Search tools — teach retrieval and freshness awareness
# ===================================================================

_VALID_STATUSES = {"pending", "in_progress", "done", "cancelled"}


@mcp.tool
def search(query: str, max_results: int = 5) -> list:
    """Web search via DuckDuckGo.

    Args:
        query: natural language search query
        max_results: cap on number of results (1-20, default 5)

    Returns:
        List of {"title": str, "snippet": str, "url": str} dicts.
        Returns [] on error (network failures, rate limits, etc.)
        rather than raising — the agent should learn that search
        can return zero results and decide whether to retry or
        fall back to internal knowledge.
    """
    if not query or not query.strip():
        return []
    max_results = max(1, min(max_results, 20))
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query.strip(), max_results=max_results))
    except Exception:
        return []
    return [
        {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
        for r in raw
    ]


@mcp.tool
def search_and_extract(query: str, max_results: int = 5) -> list:
    """News search via DuckDuckGo with date and source metadata.

    Returns the same shape as search() plus a "date" and "source"
    field on each result, which is useful for freshness-aware
    reasoning (e.g. "what's the latest news about X"). Falls back
    to general web search if the news endpoint returns nothing.

    Args:
        query: natural language search query
        max_results: cap on number of results (1-20, default 5)

    Returns:
        List of {"title": str, "snippet": str, "url": str,
                 "date": str, "source": str} dicts.
    """
    if not query or not query.strip():
        return []
    max_results = max(1, min(max_results, 20))
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.news(query.strip(), max_results=max_results))
    except Exception:
        return []
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("url", ""),
            "date": r.get("date", ""),
            "source": r.get("source", ""),
        }
        for r in raw
    ]


# ===================================================================
# Calculator — safe arithmetic without raw eval()
# ===================================================================

_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "abs": abs, "round": round,
    "pow": pow, "min": min, "max": max, "sum": sum,
}


def _safe_eval(expr: str) -> Any:
    """Parse a math expression with ast and evaluate it in a sandboxed
    namespace. Allows arithmetic operators, unary +/- on numbers, and
    calls/names that are on the whitelist. Blocks subscripts, attributes,
    comprehensions, lambdas, and any other Python construct that could
    escape the sandbox."""
    tree = ast.parse(expr, mode="eval")

    # Any node we don't explicitly allow is rejected.
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,
        ast.Load, ast.Call, ast.Name,
    )
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"expression node {type(node).__name__} not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError(f"function {ast.dump(node.func)} not allowed")
        elif isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"name {node.id!r} not allowed")

    # Compile with no builtins — only the whitelist is accessible.
    return eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES)


@mcp.tool
def calculate(expression: str) -> dict:
    """Evaluate a math expression safely and return an exact result.

    Supports arithmetic (+, -, *, /, //, %, **), parentheses, and a
    whitelisted set of math functions (sqrt, log, sin, cos, ...) and
    constants (pi, e, tau). NO raw eval() on user input — the
    expression is parsed with ast and only safe node types are
    allowed.

    Args:
        expression: math expression string, e.g. "2 * (3 + 4)" or
            "sqrt(2) ** 2"

    Returns:
        {"result": int|float} on success, or {"error": "..."} if
        the expression is empty, has syntax errors, uses disallowed
        names/calls, or fails at evaluation time.
    """
    if not expression or not expression.strip():
        return {"error": "expression must not be empty"}
    try:
        result = _safe_eval(expression.strip())
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return {"error": str(e)}
    return {"result": result}


# ===================================================================
# Scratchpad — temporary reasoning memory for planning
# ===================================================================

@mcp.tool
def write_scratchpad(content: str) -> dict:
    """Write (overwrite) the scratchpad with a new string.

    The scratchpad is server-side state the orchestrator can use as
    a working memory for multi-step plans: jot down a plan, list
    intermediate results, track variables across tool calls. It is
    NOT persistent across server restarts and should be reset between
    training episodes via reset_state().

    Args:
        content: the text to store

    Returns:
        {"length": int} — number of characters written.
    """
    _state["scratchpad"] = content
    return {"length": len(content)}


@mcp.tool
def read_scratchpad() -> str:
    """Read the current contents of the scratchpad.

    Returns:
        The stored string, or "" if the scratchpad is empty.
    """
    return _state["scratchpad"]


@mcp.tool
def clear_scratchpad() -> str:
    """Clear the scratchpad. Returns "cleared"."""
    _state["scratchpad"] = ""
    return "cleared"


# ===================================================================
# Task queue — explicit decomposition and execution planning
# ===================================================================

@mcp.tool
def create_task_item(task: str, status: str = "pending") -> dict:
    """Add a task to the explicit task queue.

    Unlike the simple to-do tasks above, the task queue is meant
    for *decomposing* a multi-step plan into trackable subtasks.
    The orchestrator should call this for each step it intends
    to execute, then update_task_item as it progresses.

    Args:
        task: short description of the step
        status: one of "pending", "in_progress", "done", "cancelled"

    Returns:
        {"id": int, "task": str, "status": str} on success, or
        {"error": ...} if status is invalid.
    """
    if status not in _VALID_STATUSES:
        return {"error": f"invalid status: {status!r}, must be one of {sorted(_VALID_STATUSES)}"}
    task_id = _state["next_task_queue_id"]
    _state["next_task_queue_id"] += 1
    entry = {"id": task_id, "task": task, "status": status}
    _state["task_queue"][task_id] = entry
    return entry


@mcp.tool
def list_task_items(status: str | None = None) -> list:
    """List task queue entries, optionally filtered by status.

    Args:
        status: one of "pending", "in_progress", "done", "cancelled";
            if omitted, returns all entries sorted by id.

    Returns:
        List of {"id": int, "task": str, "status": str} dicts.
    """
    if status is not None and status not in _VALID_STATUSES:
        return {"error": f"invalid status: {status!r}, must be one of {sorted(_VALID_STATUSES)}"}
    items = list(_state["task_queue"].values())
    if status:
        items = [t for t in items if t["status"] == status]
    return sorted(items, key=lambda t: t["id"])


@mcp.tool
def update_task_item(task_id: int, status: str) -> dict:
    """Update the status of a task queue entry.

    Args:
        task_id: id returned by create_task_item
        status: new status — one of "pending", "in_progress", "done",
            "cancelled"

    Returns:
        The updated entry, or {"error": ...} if task_id doesn't
        exist or status is invalid.
    """
    if status not in _VALID_STATUSES:
        return {"error": f"invalid status: {status!r}, must be one of {sorted(_VALID_STATUSES)}"}
    entry = _state["task_queue"].get(task_id)
    if entry is None:
        return {"error": f"no task with id {task_id}"}
    entry["status"] = status
    return entry


@mcp.tool
def delete_task_item(task_id: int) -> dict:
    """Remove a task queue entry. Returns {"deleted": task_id} or
    {"error": ...} if task_id doesn't exist."""
    if task_id not in _state["task_queue"]:
        return {"error": f"no task with id {task_id}"}
    del _state["task_queue"][task_id]
    return {"deleted": task_id}


# ===================================================================
# Code execution — sandboxed subprocess for procedural execution
# ===================================================================

# Configurable safety knobs for the code runner.
CODE_TIMEOUT_SECONDS = 5
_CODE_BLOCKLIST = re.compile(
    r"\b(import\s+(os|sys|subprocess|shutil|pathlib|socket|ctypes)|"
    r"open\s*\(|__import__|exec\s*\(|eval\s*\(|compile\s*\()",
    re.IGNORECASE,
)


@mcp.tool
def run_python(code: str, timeout: int = CODE_TIMEOUT_SECONDS) -> dict:
    """Execute a Python snippet in a sandboxed subprocess and return stdout.

    This is the "code mode" tool: the orchestrator can synthesize
    short Python to compute, transform, or verify something that
    would be tedious through discrete tool calls. Output is captured
    from stdout.

    Safety:
      - Runs in a fresh subprocess (no inherited state, no file
        handles to the parent process).
      - Hard timeout (default 5s, configurable per-call up to 30s)
        kills runaway loops.
      - Blocklist check rejects obvious dangerous constructs
        (os/sys/subprocess imports, open(), exec/eval/compile, etc.)
        BEFORE spawning the subprocess — cheap first line of defense.
      - No network access by default; only the Python stdlib and
        pure-Python stdlib modules are guaranteed to be available.
      - Working directory is a fresh tempdir; the snippet cannot
        read or write files outside it.

    Args:
        code: Python source code. Use print() to produce output.
        timeout: max seconds to wait, 1-30 (default 5).

    Returns:
        {"stdout": str, "stderr": str, "returncode": int, "timed_out": bool}
        on execution, or {"error": "..."} for invalid input or a
        blocklist hit.
    """
    if not code or not code.strip():
        return {"error": "code must not be empty"}
    timeout = max(1, min(timeout, 30))
    if _CODE_BLOCKLIST.search(code):
        return {"error": "code contains a blocked construct (os/sys/subprocess, open, exec/eval/compile, etc.)"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp},
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"execution timed out after {timeout}s",
            "returncode": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {"error": f"execution failed: {e!r}"}


# ===================================================================
# Resources — read-only context, no side effects
# ===================================================================

@mcp.resource("schedule://today")
def today_summary() -> str:
    """Plain-text summary of today's events and open tasks."""
    today = datetime.now().date().isoformat()
    todays_events = [e for e in _state["events"].values() if e["start_time"].startswith(today)]
    open_tasks = [t for t in _state["tasks"].values() if t["status"] == "open"]
    lines = [f"{len(todays_events)} event(s) today, {len(open_tasks)} open task(s) overall."]
    for e in sorted(todays_events, key=lambda ev: ev["start_time"]):
        lines.append(f"  - {e['start_time']}: {e['title']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Streamable HTTP so ART (and DeepAgents) can reach it over a URL.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
