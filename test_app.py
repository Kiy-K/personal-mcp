"""
Smoke tests for the concierge MCP server.

Run: pytest test_app.py -v
"""
import pytest

import server as app


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset server state before every test so tests are independent."""
    app.reset_state()
    yield
    app.reset_state()


# -----------------------------------------------------------------
# Existing tools (regression tests)
# -----------------------------------------------------------------

def test_reset_state():
    app.create_event("a", "2026-06-22T10:00:00", "2026-06-22T11:00:00")
    app.create_task("t")
    app.create_note("n", "c")
    result = app.reset_state()
    assert result == "state reset"
    assert app.list_events() == []
    assert app.list_tasks() == []


def test_create_event_validates_times():
    assert "error" in app.create_event("a", "not-a-date", "2026-06-22T11:00:00")
    assert "error" in app.create_event("a", "2026-06-22T11:00:00", "2026-06-22T10:00:00")


def test_cancel_requires_confirm():
    e = app.create_event("a", "2026-06-22T10:00:00", "2026-06-22T11:00:00")
    result = app.cancel_event(e["id"], confirm=False)
    assert result.get("requires_confirmation") is True
    # Event should still exist
    assert len(app.list_events()) == 1
    # Now actually cancel
    result = app.cancel_event(e["id"], confirm=True)
    assert result == {"cancelled": e["id"]}
    assert app.list_events() == []


# -----------------------------------------------------------------
# Search
# -----------------------------------------------------------------

def test_search_empty_query():
    assert app.search("") == []
    assert app.search("   ") == []


def test_search_returns_structured():
    results = app.search("python tutorial", max_results=2)
    assert isinstance(results, list)
    if results:  # may be empty if network is down
        r = results[0]
        assert set(r.keys()) == {"title", "snippet", "url"}


def test_search_and_extract_returns_news_shape():
    results = app.search_and_extract("python release", max_results=2)
    assert isinstance(results, list)
    if results:
        r = results[0]
        # News results have date and source; general search doesn't
        assert "date" in r
        assert "source" in r


# -----------------------------------------------------------------
# Calculator
# -----------------------------------------------------------------

def test_calculate_basic_arithmetic():
    assert app.calculate("2 + 2") == {"result": 4}
    assert app.calculate("2 * (3 + 4)") == {"result": 14}
    assert app.calculate("10 / 4") == {"result": 2.5}
    assert app.calculate("2 ** 10") == {"result": 1024}


def test_calculate_constants_and_functions():
    assert app.calculate("sqrt(16)")["result"] == 4.0
    assert app.calculate("pi")["result"] == pytest.approx(3.14159, rel=1e-3)
    assert app.calculate("sin(0)")["result"] == 0.0


def test_calculate_blocks_unsafe():
    assert "error" in app.calculate("")
    assert "error" in app.calculate("__import__('os')")
    assert "error" in app.calculate("open('/etc/passwd')")
    assert "error" in app.calculate("(1).__class__")
    assert "error" in app.calculate("x + 1")  # undefined name
    assert "error" in app.calculate("1/0")  # zero division


# -----------------------------------------------------------------
# Scratchpad
# -----------------------------------------------------------------

def test_scratchpad_write_read_clear():
    assert app.write_scratchpad("hello") == {"length": 5}
    assert app.read_scratchpad() == "hello"
    assert app.clear_scratchpad() == "cleared"
    assert app.read_scratchpad() == ""


def test_scratchpad_overwrite():
    app.write_scratchpad("first")
    app.write_scratchpad("second longer")
    assert app.read_scratchpad() == "second longer"
    assert app.write_scratchpad("") == {"length": 0}


# -----------------------------------------------------------------
# Task queue
# -----------------------------------------------------------------

def test_task_queue_lifecycle():
    t1 = app.create_task_item("step 1")
    t2 = app.create_task_item("step 2", "in_progress")
    assert t1["status"] == "pending"
    assert t2["status"] == "in_progress"
    assert len(app.list_task_items()) == 2
    updated = app.update_task_item(t1["id"], "done")
    assert updated["status"] == "done"
    assert len(app.list_task_items(status="done")) == 1
    assert app.delete_task_item(t1["id"]) == {"deleted": t1["id"]}
    assert len(app.list_task_items()) == 1


def test_task_queue_validates_status():
    assert "error" in app.create_task_item("x", status="invalid")
    t = app.create_task_item("y")
    assert "error" in app.update_task_item(t["id"], "invalid")


def test_task_queue_validates_id():
    assert "error" in app.update_task_item(999, "done")
    assert "error" in app.delete_task_item(999)


# -----------------------------------------------------------------
# Code execution
# -----------------------------------------------------------------

def test_run_python_basic():
    r = app.run_python("print('hello')")
    assert r["stdout"].strip() == "hello"
    assert r["returncode"] == 0
    assert r["timed_out"] is False


def test_run_python_captures_stderr():
    # Trigger a NameError which naturally writes a traceback to stderr
    # (we can't use "import sys" because the blocklist correctly blocks it)
    r = app.run_python("undefined_variable_xyz")
    assert r["returncode"] != 0
    assert "NameError" in r["stderr"]


def test_run_python_returns_nonzero_on_exception():
    r = app.run_python("raise ValueError('boom')")
    assert r["returncode"] != 0
    assert "ValueError" in r["stderr"]


def test_run_python_empty_rejected():
    assert "error" in app.run_python("")
    assert "error" in app.run_python("   ")


def test_run_python_blocks_dangerous_imports():
    # pydantic-monty: dangerous imports are unreachable in the interpreter
    # itself. `import os` is a no-op (returns 0), `open()` raises
    # NameError because the name doesn't exist. The orchestrator
    # should learn to handle these as execution failures, not to
    # expect a pre-flight "blocked construct" error.
    assert "error" in app.run_python("")  # empty is the only pre-flight error
    r = app.run_python("import os")
    # import os is silently allowed but harmless (no actual os module)
    assert r["returncode"] == 0
    r = app.run_python("open('/etc/passwd')")
    assert r["returncode"] != 0  # NameError: open is not defined
    r = app.run_python("exec('print(1)')")
    assert r["returncode"] != 0  # NameError: exec is not defined
    r = app.run_python("eval('1+1')")
    assert r["returncode"] != 0


def test_run_python_timeout():
    r = app.run_python("while True: pass", timeout=1)
    assert r["timed_out"] is True
    assert r["returncode"] == -1


def test_run_python_timeout_clamped():
    # timeout > 30 should be clamped to 30
    r = app.run_python("print(1)", timeout=999)
    assert r["returncode"] == 0


# -----------------------------------------------------------------
# Transforms / server
# -----------------------------------------------------------------

def test_server_has_no_transforms_by_default():
    """By default, all tools are exposed directly (no search proxy,
    no tag transform). CodeMode is opt-in via env var."""
    assert app.mcp.name == "concierge-assistant"
    assert app.mcp._transforms == []


def test_tools_directly_callable():
    """All registered tools should be reachable by direct call —
    not hidden behind a search/call proxy."""
    # Just verify several tools are registered and callable
    for tool_name in [
        "create_event", "list_events", "create_task", "complete_task",
        "create_note", "search_notes", "search", "search_and_extract",
        "calculate", "write_scratchpad", "read_scratchpad",
        "create_task_item", "list_task_items", "update_task_item",
        "delete_task_item", "run_python", "current_time", "set_budget",
        "get_action_count", "get_trajectory", "fetch_url", "extract_json",
        "reset_state",
    ]:
        assert callable(getattr(app, tool_name)), f"{tool_name} not registered"


# -----------------------------------------------------------------
# Meta tools — current_time, action budget, trajectory
# -----------------------------------------------------------------

def test_current_time_utc():
    r = app.current_time("UTC")
    assert "iso" in r
    assert "timezone" in r
    assert "unix" in r
    assert r["timezone"] == "UTC"
    assert r["iso"].endswith("+00:00")


def test_current_time_named_tz():
    r = app.current_time("Asia/Ho_Chi_Minh")
    assert r["timezone"] == "Asia/Ho_Chi_Minh"
    # +07:00 offset
    assert "+07:00" in r["iso"]


def test_current_time_bad_tz():
    r = app.current_time("Mars/Olympus")
    assert "error" in r


def test_action_count_starts_at_zero():
    r = app.get_action_count()
    assert r["action_count"] == 0
    assert r["budget"] is None
    assert r["remaining"] is None


def test_set_budget_enforces_cap():
    app.set_budget(2)
    r = app.create_task("t1")  # mutating, counts
    assert r["id"] == 1
    app.create_task("t2")  # mutating, counts
    # Third mutating call should fail
    r = app.create_task("t3")
    assert "error" in r
    assert "budget exhausted" in r["error"]
    # Read-only tools should still work
    r = app.list_tasks()
    assert isinstance(r, list)


def test_set_budget_zero_clears():
    # set_budget is exempt from counting, so action_count stays at 0
    # after set_budget(1). t1 -> action_count=1 (within budget), t2
    # would be action_count=2 (exceeds), so blocked. set_budget(0)
    # clears the budget (also exempt), so t3 succeeds with id 2.
    app.set_budget(1)
    r1 = app.create_task("t1")
    assert r1["id"] == 1
    r2 = app.create_task("t2")
    assert "error" in r2  # budget exhausted
    app.set_budget(0)  # clears budget
    r3 = app.create_task("t3")
    assert r3["id"] == 2  # t1 was id 1, t2 was blocked, t3 gets id 2


def test_get_trajectory_records_tool_calls():
    app.create_task("trajectory-test")
    app.calculate("1 + 1")
    traj = app.get_trajectory()
    # The fixture calls reset_state first, so trajectory has at least 3 entries
    assert len(traj) >= 3
    tool_names = [e["tool"] for e in traj]
    assert "create_task" in tool_names
    assert "calculate" in tool_names
    # Each entry has the required shape
    for entry in traj:
        assert "step" in entry
        assert "tool" in entry
        assert "arguments" in entry
        assert "result" in entry


def test_get_trajectory_last_n():
    app.create_task("a")
    app.create_task("b")
    app.create_task("c")
    last_two = app.get_trajectory(last_n=2)
    assert len(last_two) == 2
    # Returned most-recent-first
    assert last_two[0]["tool"] == "create_task"
    assert last_two[1]["tool"] == "create_task"


# -----------------------------------------------------------------
# fetch_url
# -----------------------------------------------------------------

def test_fetch_url_empty():
    r = app.fetch_url("")
    assert "error" in r


def test_fetch_url_bad_scheme():
    r = app.fetch_url("ftp://example.com")
    assert "error" in r


def test_fetch_url_real():
    """Hit a real URL (httpbin.org /example endpoint) to verify the
    fetch path works end-to-end. Skipped if no network."""
    r = app.fetch_url("https://example.com", max_bytes=10000, timeout=10)
    if "error" in r and "fetch failed" in r["error"]:
        pytest.skip(f"no network: {r['error']}")
    assert r["status"] == 200
    assert "Example Domain" in r["text"] or "example" in r["text"].lower()


# -----------------------------------------------------------------
# extract_json
# -----------------------------------------------------------------

def test_extract_json_fenced():
    r = app.extract_json('```json\n{"a": 1, "b": [2, 3]}\n```')
    assert r["json"] == {"a": 1, "b": [2, 3]}


def test_extract_json_bare_object():
    r = app.extract_json('Here is the data: {"x": 10, "y": 20}')
    assert r["json"] == {"x": 10, "y": 20}


def test_extract_json_array():
    r = app.extract_json("results: [1, 2, 3]")
    assert r["json"] == [1, 2, 3]


def test_extract_json_nested():
    r = app.extract_json('```\n{"outer": {"inner": [1, 2]}}\n```')
    assert r["json"] == {"outer": {"inner": [1, 2]}}


def test_extract_json_no_json():
    r = app.extract_json("this text has no structured data")
    assert "error" in r


def test_extract_json_empty():
    r = app.extract_json("")
    assert "error" in r
