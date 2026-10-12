"""
Tests for AgentMind
"""

import pytest
from datetime import datetime, timedelta
from agentmind import Memory, MemoryConfig


@pytest.fixture
def memory():
    """Create test memory"""
    config = MemoryConfig(storage="sqlite", db_path=":memory:")
    return Memory(config=config)


def test_remember_and_recall(memory):
    """Test saving and retrieving memories"""
    # Save
    memory.remember(
        agent_id="test-agent",
        content="User likes Python programming",
        metadata={"category": "preference"}
    )
    
    # Retrieve
    memories = memory.recall(
        agent_id="test-agent",
        query="programming preferences",
        limit=5
    )
    
    assert len(memories) > 0
    assert "Python" in memories[0].content


def test_recall_with_filters(memory):
    """Test retrieval with filters"""
    # Save several memories
    memory.remember("agent-1", "Memory 1", metadata={"type": "A"})
    memory.remember("agent-1", "Memory 2", metadata={"type": "B"})
    memory.remember("agent-1", "Memory 3", metadata={"type": "A"})
    
    # Retrieve with filter
    memories = memory.recall(
        agent_id="agent-1",
        filters={"type": "A"},
        limit=10
    )
    
    assert len(memories) == 2
    assert all(m.metadata["type"] == "A" for m in memories)


def test_recall_with_time_range(memory):
    """Test retrieval with time range"""
    now = datetime.now()
    
    # Save with different times
    memory.remember(
        "agent-1",
        "Old memory",
        timestamp=now - timedelta(days=10)
    )
    
    memory.remember(
        "agent-1",
        "Recent memory",
        timestamp=now - timedelta(days=1)
    )
    
    # Retrieve only recent
    memories = memory.recall(
        agent_id="agent-1",
        time_range=(now - timedelta(days=5), now),
        limit=10
    )
    
    assert len(memories) == 1
    assert "Recent" in memories[0].content


def test_forget(memory):
    """Test deleting memories"""
    # Save
    mem_id = memory.remember("agent-1", "Test memory")
    
    # Check exists
    memories = memory.recall("agent-1", limit=10)
    assert len(memories) == 1
    
    # Delete
    deleted = memory.forget("agent-1", memory_ids=[mem_id])
    assert deleted == 1
    
    # Check deleted
    memories = memory.recall("agent-1", limit=10)
    assert len(memories) == 0


def test_get_stats(memory):
    """Test getting statistics"""
    # Save several memories
    memory.remember("agent-1", "Memory 1")
    memory.remember("agent-1", "Memory 2")
    memory.remember("agent-1", "Memory 3")
    
    # Get stats
    stats = memory.get_stats("agent-1")
    
    assert stats["total_memories"] == 3
    assert "oldest_memory" in stats
    assert "newest_memory" in stats


def test_semantic_search(memory):
    """Test semantic search"""
    # Save memories
    memory.remember("agent-1", "I love playing basketball and soccer")
    memory.remember("agent-1", "Python is my favorite programming language")
    memory.remember("agent-1", "I enjoy hiking in the mountains")
    
    # Search by meaning
    memories = memory.recall(
        agent_id="agent-1",
        query="sports and outdoor activities",
        limit=5
    )
    
    # Should find memories
    assert len(memories) > 0
    
    # Check that sports memories have high score
    sports_keywords = ["basketball", "soccer", "hiking"]
    found_sports = any(
        any(keyword in mem.content.lower() for keyword in sports_keywords)
        for mem in memories[:2]
    )
    assert found_sports


def test_importance(memory):
    """Test memory importance"""
    # Save with different importance
    memory.remember("agent-1", "Low importance", importance=0.1)
    memory.remember("agent-1", "High importance", importance=0.9)
    
    # Get all
    memories = memory.recall("agent-1", limit=10)
    
    # Check importance saved
    assert any(m.importance == 0.1 for m in memories)
    assert any(m.importance == 0.9 for m in memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
