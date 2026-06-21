"""
Smoke tests for the concierge MCP server.

Run: pytest test_app.py -v
"""
import pytest

import app


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
    assert "error" in app.run_python("import os")
    assert "error" in app.run_python("import subprocess")
    assert "error" in app.run_python("import socket")
    assert "error" in app.run_python("open('/etc/passwd')")
    assert "error" in app.run_python("exec('print(1)')")
    assert "error" in app.run_python("eval('1+1')")


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

def test_server_has_transforms():
    """The default config should have ToolTransform + BM25SearchTransform."""
    assert app.mcp.name == "concierge-assistant"
    assert len(app.mcp._transforms) == 2


def test_tool_tags_applied():
    """The ToolTransform should have added tags to registered tools."""
    tag_transform = app.mcp._transforms[0]
    # ToolTransform stores configs in its _transforms dict
    calc_config = tag_transform._transforms.get("calculate")
    assert calc_config is not None
    assert "math" in calc_config.tags
    assert calc_config.title == "Calculate Math Expression"

    cancel_config = tag_transform._transforms.get("cancel_event")
    assert "destructive" in cancel_config.tags
