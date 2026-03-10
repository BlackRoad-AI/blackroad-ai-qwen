"""
Tests for memory-bridge/memory_integration.py.

Validates graceful failure paths when the memory shell script is missing
or returns a non-zero exit code.
"""
import os
import sys
import pytest

# Ensure the memory-bridge package is importable when tests run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory-bridge"))

from memory_integration import MemoryBridge


# ---------------------------------------------------------------------------
# connect() — graceful failure when script is absent
# ---------------------------------------------------------------------------
class TestMemoryBridgeConnect:
    @pytest.mark.asyncio
    async def test_connect_fails_gracefully_with_missing_script(self):
        bridge = MemoryBridge()
        bridge.memory_script = "/nonexistent/path/memory-system.sh"
        result = await bridge.connect()
        assert result is False
        assert bridge.is_connected() is False

    @pytest.mark.asyncio
    async def test_connect_sets_connected_false_on_error(self):
        bridge = MemoryBridge()
        bridge.memory_script = "/nonexistent/path/memory-system.sh"
        await bridge.connect()
        assert bridge.connected is False


# ---------------------------------------------------------------------------
# is_connected()
# ---------------------------------------------------------------------------
class TestMemoryBridgeIsConnected:
    def test_not_connected_by_default(self):
        bridge = MemoryBridge()
        assert bridge.is_connected() is False

    @pytest.mark.asyncio
    async def test_disconnected_after_disconnect(self):
        bridge = MemoryBridge()
        bridge.connected = True  # simulate connected state
        await bridge.disconnect()
        assert bridge.is_connected() is False


# ---------------------------------------------------------------------------
# get_context() — returns None when not connected
# ---------------------------------------------------------------------------
class TestMemoryBridgeGetContext:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_connected(self):
        bridge = MemoryBridge()
        result = await bridge.get_context("session-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_with_no_session_id(self):
        bridge = MemoryBridge()
        result = await bridge.get_context(None)
        assert result is None


# ---------------------------------------------------------------------------
# save_interaction() — no-op when not connected (no exception raised)
# ---------------------------------------------------------------------------
class TestMemoryBridgeSaveInteraction:
    @pytest.mark.asyncio
    async def test_save_does_not_raise_when_not_connected(self):
        bridge = MemoryBridge()
        # Should complete without raising
        await bridge.save_interaction(
            session_id="test-session",
            user_message="hello",
            assistant_response="hi there",
        )

    @pytest.mark.asyncio
    async def test_save_with_tags_does_not_raise_when_not_connected(self):
        bridge = MemoryBridge()
        await bridge.save_interaction(
            session_id="test-session",
            user_message="hello",
            assistant_response="hi there",
            tags=["tag1", "tag2"],
        )


# ---------------------------------------------------------------------------
# broadcast_status() — no-op when not connected
# ---------------------------------------------------------------------------
class TestMemoryBridgeBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_does_not_raise_when_not_connected(self):
        bridge = MemoryBridge()
        await bridge.broadcast_status("online", {"version": "1.0"})


# ---------------------------------------------------------------------------
# get_collaboration_context() — returns empty dict when not connected
# ---------------------------------------------------------------------------
class TestMemoryBridgeCollaboration:
    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_not_connected(self):
        bridge = MemoryBridge()
        result = await bridge.get_collaboration_context()
        assert result == {}


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
class TestGetMemoryBridge:
    def test_singleton_returns_memory_bridge_instance(self):
        from memory_integration import get_memory_bridge
        bridge = get_memory_bridge()
        assert isinstance(bridge, MemoryBridge)

    def test_singleton_same_instance(self):
        from memory_integration import get_memory_bridge
        a = get_memory_bridge()
        b = get_memory_bridge()
        assert a is b
