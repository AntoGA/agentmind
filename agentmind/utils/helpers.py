"""
Utility Functions
"""

from typing import List
import numpy as np

from agentmind.core.store import MemoryRecord


def format_memories(memories: List[MemoryRecord], include_score: bool = True) -> str:
    """
    Format memories into readable string.
    
    Args:
        memories: List of memories
        include_score: Include score in output
    
    Returns:
        Formatted string
    
    Example:
        >>> memories = memory.recall("agent-1", "Python")
        >>> print(format_memories(memories))
        - User prefers Python (score: 0.95)
        - User is a Python developer (score: 0.87)
    """
    if not memories:
        return "No memories found"
    
    lines = []
    for mem in memories:
        if include_score and mem.score is not None:
            lines.append(f"- {mem.content} (score: {mem.score:.2f})")
        else:
            lines.append(f"- {mem.content}")
    
    return "\n".join(lines)


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity (0.0 - 1.0)
    
    Example:
        >>> similarity = calculate_similarity([1, 0, 0], [1, 0, 0])
        >>> print(similarity)  # 1.0
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))
