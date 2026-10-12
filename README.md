# 🧠 AgentMind — AI Agent Memory Layer

[![PyPI version](https://badge.fury.io/py/agentmind.svg)](https://badge.fury.io/py/agentmind)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Persistent memory for AI agents that actually works.**

AgentMind solves a critical problem with AI agents: they forget everything between sessions. This library provides long-term memory with semantic search, temporal context, and integration with popular frameworks.

## ✨ Features

- 🧠 **Long-term memory** — agents remember context between sessions
- 🔍 **Semantic search** — find relevant memories by meaning
- ⏰ **Temporal context** — understand "when" events occurred
- 👤 **User preferences** — personalize for each user
- 🔗 **Integrations** — LangChain, AutoGen, CrewAI, OpenAI
- 🚀 **Simple API** — 3 lines of code to add memory
- 💾 **Flexible storage** — SQLite, PostgreSQL, Redis, ChromaDB

## 🚀 Quick Start

### Installation

```bash
pip install agentmind
```

### Basic Usage

```python
from agentmind import Memory

# Initialize memory
memory = Memory(storage="sqlite", path="agent_memory.db")

# Remember
memory.remember(
    agent_id="assistant-1",
    content="User prefers Python over JavaScript",
    metadata={"category": "preference", "user_id": "user-123"}
)

# Recall
memories = memory.recall(
    agent_id="assistant-1",
    query="What programming languages does the user like?",
    limit=5
)

for mem in memories:
    print(f"- {mem.content} (relevance: {mem.score:.2f})")
```

### LangChain Integration

```python
from agentmind.integrations import LangChainMemory
from langchain.agents import initialize_agent

# Create memory for LangChain
memory = LangChainMemory(agent_id="my-agent")

# Use in agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,  # Agent now remembers context!
    verbose=True
)

# Agent automatically saves and retrieves memories
agent.run("What did we discuss yesterday?")
```

## 📚 Examples

### 1. Personal Assistant

```python
from agentmind import Memory

memory = Memory()

# User mentions preferences
memory.remember(
    agent_id="assistant",
    content="User is a senior Python developer interested in AI/ML",
    metadata={"type": "user_profile"}
)

memory.remember(
    agent_id="assistant",
    content="User prefers concise code examples without excessive comments",
    metadata={"type": "preference"}
)

# Later, agent can personalize responses
context = memory.recall(
    agent_id="assistant",
    query="user background and preferences",
    limit=3
)

# Use context in prompt
prompt = f"""
Based on user context:
{context}

Provide a response tailored to their expertise level.
"""
```

### 2. Multi-Agent System

```python
from agentmind import Memory

# Shared memory for agent team
shared_memory = Memory(storage="postgresql", connection_string="...")

# Researcher agent saves findings
shared_memory.remember(
    agent_id="researcher",
    content="Found 3 relevant papers on transformer architectures",
    metadata={"task": "research", "topic": "transformers"}
)

# Writer agent uses these findings
research = shared_memory.recall(
    agent_id="writer",
    query="transformer architecture papers",
    filters={"task": "research"}
)

# Writer creates content based on research
```

### 3. Temporal Context

```python
from agentmind import Memory
from datetime import datetime, timedelta

memory = Memory()

# Remember with timestamp
memory.remember(
    agent_id="scheduler",
    content="Meeting with client about project requirements",
    timestamp=datetime(2026, 1, 15, 14, 0),
    metadata={"type": "event", "priority": "high"}
)

# Recall recent events
recent = memory.recall(
    agent_id="scheduler",
    query="meetings and events",
    time_range=(datetime.now() - timedelta(days=7), datetime.now())
)
```

## 💾 Storage Types

### SQLite (default, for development)

```python
memory = Memory(storage="sqlite", path="memory.db")
```

### PostgreSQL (for production)

```python
memory = Memory(
    storage="postgresql",
    connection_string="postgresql://user:pass@localhost:5432/agentmind"
)
```

### Redis (for fast access)

```python
memory = Memory(
    storage="redis",
    connection_string="redis://localhost:6379",
    ttl=86400  # 24 hours
)
```

### ChromaDB (for advanced vector search)

```python
memory = Memory(
    storage="chromadb",
    collection_name="agent_memories",
    persist_directory="./chroma_db"
)
```

## 🔧 Configuration

```python
from agentmind import Memory, MemoryConfig

config = MemoryConfig(
    storage="postgresql",
    connection_string="postgresql://...",
    embedding_model="text-embedding-3-small",  # OpenAI embeddings
    embedding_dimension=1536,
    max_memories=10000,  # Limit per agent
    ttl=2592000,  # 30 days in seconds
    auto_cleanup=True  # Automatic cleanup of old entries
)

memory = Memory(config=config)
```

## 📊 API Reference

### `Memory.remember()`

Save a memory.

```python
memory.remember(
    agent_id: str,              # Agent ID
    content: str,               # Memory text
    metadata: dict = None,      # Additional metadata
    timestamp: datetime = None, # Timestamp (default: now)
    importance: float = 1.0     # Importance (0.0 - 1.0)
)
```

### `Memory.recall()`

Retrieve relevant memories.

```python
memories = memory.recall(
    agent_id: str,                    # Agent ID
    query: str = None,                # Search query
    filters: dict = None,             # Metadata filters
    time_range: tuple = None,         # (start_time, end_time)
    limit: int = 10,                  # Max results
    min_score: float = 0.0            # Minimum relevance
)
```

### `Memory.forget()`

Delete memories.

```python
memory.forget(
    agent_id: str,
    memory_ids: list = None,    # Specific IDs
    filters: dict = None,       # Delete by filter
    older_than: datetime = None # Delete old ones
)
```

### `Memory.get_stats()`

Get memory statistics.

```python
stats = memory.get_stats(agent_id="my-agent")
# {
#     "total_memories": 1234,
#     "oldest_memory": "2026-01-01T00:00:00",
#     "newest_memory": "2026-01-15T12:34:56",
#     "storage_size_mb": 45.2
# }
```

## 🔗 Integrations

### LangChain

```python
from agentmind.integrations import LangChainMemory

memory = LangChainMemory(
    agent_id="my-agent",
    return_messages=True,
    memory_key="chat_history"
)
```

### AutoGen

```python
from agentmind.integrations import AutoGenMemory

memory = AutoGenMemory(agent_id="assistant")

# Use in AutoGen agent
assistant = autogen.AssistantAgent(
    name="assistant",
    memory=memory
)
```

### OpenAI Assistants API

```python
from agentmind.integrations import OpenAIMemory

memory = OpenAIMemory(
    agent_id="my-assistant",
    thread_id="thread-123"
)

# Sync with OpenAI thread
memory.sync_with_thread()
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=agentmind tests/

# Integration tests only
pytest tests/integration/
```

## 📈 Roadmap

- [x] v0.1.0 — Basic functionality
- [x] v0.2.0 — LangChain integration
- [ ] v0.3.0 — AutoGen and CrewAI integration
- [ ] v0.4.0 — Web UI for memory management
- [ ] v0.5.0 — Distributed memory (Redis Cluster)
- [ ] v1.0.0 — Production-ready release

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

Inspired by:
- [Mem0](https://github.com/mem0ai/mem0) — original persistent memory idea
- [LangChain](https://github.com/langchain-ai/langchain) — LLM integrations
- [ChromaDB](https://github.com/chroma-core/chroma) — vector databases

## 📧 Contact

- GitHub: [@AntoGA](https://github.com/AntoGA)
- Issues: [GitHub Issues](https://github.com/AntoGA/agentmind/issues)

---

**⭐ If this project is useful, please star it! It helps with development.**
