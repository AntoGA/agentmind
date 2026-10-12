"""
Basic Usage Example
Simple example of using AgentMind.
"""

from agentmind import Memory

# Initialize memory
memory = Memory(storage="sqlite", db_path="example_memory.db")

# Clear previous data (for example)
memory.forget("assistant-1")

print("🧠 AgentMind - Basic Usage Example\n")

# 1. Save memories
print("📝 Saving memories...")

memory.remember(
    agent_id="assistant-1",
    content="User is a senior Python developer with 10 years of experience",
    metadata={"category": "profile", "user_id": "user-123"}
)

memory.remember(
    agent_id="assistant-1",
    content="User prefers concise code examples without excessive comments",
    metadata={"category": "preference", "user_id": "user-123"}
)

memory.remember(
    agent_id="assistant-1",
    content="User is interested in AI/ML and natural language processing",
    metadata={"category": "interest", "user_id": "user-123"}
)

memory.remember(
    agent_id="assistant-1",
    content="User works at a startup focused on healthcare technology",
    metadata={"category": "work", "user_id": "user-123"}
)

print("✅ Saved 4 memories\n")

# 2. Retrieve memories
print("🔍 Recalling memories...\n")

# Search by query
memories = memory.recall(
    agent_id="assistant-1",
    query="What is the user's technical background?",
    limit=3
)

print("Query: 'What is the user's technical background?'")
print("-" * 60)
for mem in memories:
    print(f"• {mem.content}")
    print(f"  Score: {mem.score:.3f}")
    print()

# 3. Search with filters
print("🔍 Searching with filters...\n")

memories = memory.recall(
    agent_id="assistant-1",
    query="user preferences",
    filters={"category": "preference"},
    limit=5
)

print("Query: 'user preferences' (category=preference)")
print("-" * 60)
for mem in memories:
    print(f"• {mem.content}")
    print()

# 4. Statistics
print("📊 Memory Statistics")
print("-" * 60)
stats = memory.get_stats("assistant-1")
for key, value in stats.items():
    print(f"{key}: {value}")

print("\n✅ Example completed!")
