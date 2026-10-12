"""
Core Memory Implementation
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import numpy as np

from agentmind.core.store import MemoryStore, MemoryRecord
from agentmind.core.embeddings import EmbeddingModel


class MemoryConfig(BaseModel):
    """Configuration for Memory"""
    storage: str = "sqlite"
    db_path: str = "./agent_memory.db"
    embedding_model: str = "text-embedding-3-small"
    max_memories: int = 10000
    auto_cleanup: bool = True


class Memory:
    """
    Main class for working with agent memory.
    
    Example:
        >>> memory = Memory()
        >>> memory.remember("agent-1", "User prefers Python")
        >>> memories = memory.recall("agent-1", "programming preferences")
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, **kwargs):
        """
        Initialize memory.
        
        Args:
            config: Memory configuration
            **kwargs: Configuration parameters (override config)
        """
        if config is None:
            config = MemoryConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.store = MemoryStore(config)
        self.embeddings = EmbeddingModel(config.embedding_model)
    
    def remember(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        importance: float = 1.0
    ) -> str:
        """
        Save a memory.
        
        Args:
            agent_id: Agent ID
            content: Memory text
            metadata: Additional metadata
            timestamp: Timestamp (default: now)
            importance: Importance (0.0 - 1.0)
        
        Returns:
            ID of saved memory
        
        Example:
            >>> memory.remember("agent-1", "User prefers Python", metadata={"category": "preference"})
            'mem_123abc'
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        # Create embedding
        embedding = self.embeddings.encode(content)
        
        # Create record
        record = MemoryRecord(
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            timestamp=timestamp,
            importance=importance
        )
        
        # Save
        memory_id = self.store.save(record)
        
        # Auto cleanup if enabled
        if self.config.auto_cleanup:
            self._cleanup_if_needed(agent_id)
        
        return memory_id
    
    def recall(
        self,
        agent_id: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        time_range: Optional[tuple] = None,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[MemoryRecord]:
        """
        Retrieve relevant memories.
        
        Args:
            agent_id: Agent ID
            query: Search query (if None, returns latest)
            filters: Metadata filters
            time_range: Time range (start_time, end_time)
            limit: Max results
            min_score: Minimum relevance (0.0 - 1.0)
        
        Returns:
            List of relevant memories
        
        Example:
            >>> memories = memory.recall("agent-1", "Python programming", limit=5)
            >>> for mem in memories:
            ...     print(f"{mem.content} (score: {mem.score:.2f})")
        """
        # If query exists, create embedding
        query_embedding = None
        if query:
            query_embedding = self.embeddings.encode(query)
        
        # Search
        memories = self.store.search(
            agent_id=agent_id,
            query_embedding=query_embedding,
            filters=filters,
            time_range=time_range,
            limit=limit,
            min_score=min_score
        )
        
        return memories
    
    def forget(
        self,
        agent_id: str,
        memory_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete memories.
        
        Args:
            agent_id: Agent ID
            memory_ids: Specific IDs to delete
            filters: Delete by metadata filter
            older_than: Delete memories older than date
        
        Returns:
            Number of deleted memories
        
        Example:
            >>> deleted = memory.forget("agent-1", older_than=datetime(2026, 1, 1))
            >>> print(f"Deleted {deleted} memories")
        """
        return self.store.delete(
            agent_id=agent_id,
            memory_ids=memory_ids,
            filters=filters,
            older_than=older_than
        )
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent memory statistics.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Dictionary with statistics
        
        Example:
            >>> stats = memory.get_stats("agent-1")
            >>> print(f"Total memories: {stats['total_memories']}")
        """
        return self.store.get_stats(agent_id)
    
    def _cleanup_if_needed(self, agent_id: str):
        """Clean up old memories if limit exceeded"""
        stats = self.get_stats(agent_id)
        
        if stats["total_memories"] > self.config.max_memories:
            # Delete oldest and least important
            excess = stats["total_memories"] - self.config.max_memories
            
            # Get old memories
            old_memories = self.recall(
                agent_id=agent_id,
                limit=excess * 2
            )
            
            # Sort by importance and time
            old_memories.sort(key=lambda m: (m.importance, m.timestamp))
            
            # Delete excess
            to_delete = [m.id for m in old_memories[:excess]]
            self.forget(agent_id=agent_id, memory_ids=to_delete)
