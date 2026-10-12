"""
Memory Storage Backend
Abstraction for different storage systems (SQLite, PostgreSQL, Redis, ChromaDB).
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
import numpy as np

from agentmind.core.memory import MemoryConfig


class MemoryRecord(BaseModel):
    """Memory record"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = 1.0
    score: Optional[float] = None


class MemoryStore:
    """Memory storage"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.storage_type = config.storage
        
        if self.storage_type == "sqlite":
            self._init_sqlite()
        elif self.storage_type == "postgresql":
            raise NotImplementedError("PostgreSQL backend coming soon")
        elif self.storage_type == "redis":
            raise NotImplementedError("Redis backend coming soon")
        elif self.storage_type == "chromadb":
            raise NotImplementedError("ChromaDB backend coming soon")
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _init_sqlite(self):
        """Initialize SQLite"""
        self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 1.0
            )
        """)
        
        # Indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        
        self.conn.commit()
    
    def save(self, record: MemoryRecord) -> str:
        """Save memory"""
        if self.storage_type == "sqlite":
            return self._save_sqlite(record)
        else:
            raise NotImplementedError(f"Save not implemented for {self.storage_type}")
    
    def _save_sqlite(self, record: MemoryRecord) -> str:
        """Save to SQLite"""
        embedding_blob = np.array(record.embedding, dtype=np.float32).tobytes()
        metadata_json = json.dumps(record.metadata)
        timestamp_str = record.timestamp.isoformat()
        
        self.conn.execute("""
            INSERT INTO memories (id, agent_id, content, embedding, metadata, timestamp, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id,
            record.agent_id,
            record.content,
            embedding_blob,
            metadata_json,
            timestamp_str,
            record.importance
        ))
        
        self.conn.commit()
        return record.id
    
    def search(
        self,
        agent_id: str,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[MemoryRecord]:
        """Search memories"""
        if self.storage_type == "sqlite":
            return self._search_sqlite(agent_id, query_embedding, filters, time_range, limit, min_score)
        else:
            raise NotImplementedError(f"Search not implemented for {self.storage_type}")
    
    def _search_sqlite(
        self,
        agent_id: str,
        query_embedding: Optional[List[float]],
        filters: Optional[Dict[str, Any]],
        time_range: Optional[Tuple[datetime, datetime]],
        limit: int,
        min_score: float
    ) -> List[MemoryRecord]:
        """Search in SQLite"""
        # Get all agent memories
        query = "SELECT * FROM memories WHERE agent_id = ?"
        params = [agent_id]
        
        if time_range:
            start_time, end_time = time_range
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend([start_time.isoformat(), end_time.isoformat()])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit * 10)
        
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to MemoryRecord
        memories = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            timestamp = datetime.fromisoformat(row["timestamp"])
            
            record = MemoryRecord(
                id=row["id"],
                agent_id=row["agent_id"],
                content=row["content"],
                embedding=embedding,
                metadata=metadata,
                timestamp=timestamp,
                importance=row["importance"]
            )
            memories.append(record)
        
        # If query_embedding exists, compute similarity
        if query_embedding:
            query_vec = np.array(query_embedding)
            
            for mem in memories:
                mem_vec = np.array(mem.embedding)
                # Cosine similarity
                score = np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec))
                mem.score = float(score)
            
            # Sort by score
            memories.sort(key=lambda m: m.score if m.score else 0, reverse=True)
            
            # Filter by min_score
            memories = [m for m in memories if m.score and m.score >= min_score]
        
        # Apply metadata filters
        if filters:
            filtered = []
            for mem in memories:
                match = True
                for key, value in filters.items():
                    if key not in mem.metadata or mem.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered.append(mem)
            memories = filtered
        
        # Limit results
        return memories[:limit]
    
    def delete(
        self,
        agent_id: str,
        memory_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """Delete memories"""
        if self.storage_type == "sqlite":
            return self._delete_sqlite(agent_id, memory_ids, filters, older_than)
        else:
            raise NotImplementedError(f"Delete not implemented for {self.storage_type}")
    
    def _delete_sqlite(
        self,
        agent_id: str,
        memory_ids: Optional[List[str]],
        filters: Optional[Dict[str, Any]],
        older_than: Optional[datetime]
    ) -> int:
        """Delete from SQLite"""
        query = "DELETE FROM memories WHERE agent_id = ?"
        params = [agent_id]
        
        if memory_ids:
            placeholders = ",".join(["?"] * len(memory_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(memory_ids)
        
        if older_than:
            query += " AND timestamp < ?"
            params.append(older_than.isoformat())
        
        cursor = self.conn.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount
    
    def get_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics"""
        if self.storage_type == "sqlite":
            return self._get_stats_sqlite(agent_id)
        else:
            raise NotImplementedError(f"Stats not implemented for {self.storage_type}")
    
    def _get_stats_sqlite(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics from SQLite"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest
            FROM memories
            WHERE agent_id = ?
        """, (agent_id,))
        
        row = cursor.fetchone()
        
        return {
            "total_memories": row[0],
            "oldest_memory": row[1],
            "newest_memory": row[2],
            "storage_type": "sqlite"
        }
