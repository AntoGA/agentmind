"""
Multi-Agent Example
Example of using shared memory for multiple agents.
"""

from agentmind import Memory
from datetime import datetime, timedelta

print("🧠 AgentMind - Multi-Agent Example\n")

# Create shared memory for agent team
shared_memory = Memory(storage="sqlite", db_path="team_memory.db")

# Clear previous data
shared_memory.forget("researcher")
shared_memory.forget("writer")
shared_memory.forget("manager")

# 1. Researcher agent gathers information
print("🔬 Researcher Agent: Gathering information...")

shared_memory.remember(
    agent_id="researcher",
    content="Found 5 research papers on transformer architectures from 2025",
    metadata={"task": "research", "topic": "transformers", "priority": "high"}
)

shared_memory.remember(
    agent_id="researcher",
    content="Discovered new attention mechanism that improves efficiency by 30%",
    metadata={"task": "research", "topic": "attention", "priority": "high"}
)

shared_memory.remember(
    agent_id="researcher",
    content="Competitor X released similar product last month",
    metadata={"task": "research", "topic": "competitors", "priority": "medium"}
)

print("✅ Researcher saved 3 findings\n")

# 2. Writer agent uses research
print("✍️  Writer Agent: Creating content based on research...")

# Get research
research = shared_memory.recall(
    agent_id="writer",  # Writer can read researcher's memory
    query="transformer architecture research",
    filters={"task": "research"},
    limit=10
)

print(f"Writer found {len(research)} research items:")
for item in research:
    print(f"  - {item.content}")

# Writer creates content
shared_memory.remember(
    agent_id="writer",
    content="Drafted blog post about new transformer architectures (1500 words)",
    metadata={"task": "writing", "status": "draft", "word_count": 1500}
)

shared_memory.remember(
    agent_id="writer",
    content="Created technical documentation for new attention mechanism",
    metadata={"task": "writing", "status": "complete", "type": "documentation"}
)

print("✅ Writer created 2 content pieces\n")

# 3. Manager agent checks progress
print("👔 Manager Agent: Checking team progress...")

# Get all tasks
all_tasks = shared_memory.recall(
    agent_id="manager",
    query="all team activities",
    time_range=(datetime.now() - timedelta(days=1), datetime.now()),
    limit=20
)

print(f"Manager found {len(all_tasks)} recent activities:")
for task in all_tasks:
    agent = task.agent_id
    content = task.content
    task_type = task.metadata.get("task", "unknown")
    print(f"  [{agent}] {task_type}: {content}")

# Manager saves decision
shared_memory.remember(
    agent_id="manager",
    content="Approved blog post for publication. Scheduled for next Monday.",
    metadata={"task": "management", "decision": "approved"}
)

print("\n✅ Multi-agent collaboration completed!")

# 4. Show statistics
print("\n📊 Team Statistics")
print("-" * 60)

for agent in ["researcher", "writer", "manager"]:
    stats = shared_memory.get_stats(agent)
    print(f"{agent}: {stats['total_memories']} memories")
