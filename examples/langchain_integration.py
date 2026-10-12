"""
LangChain Integration Example
Example of integrating AgentMind with LangChain.
"""

import os
from dotenv import load_dotenv

from agentmind.integrations import LangChainMemory

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import initialize_agent, Tool
    from langchain.chains import ConversationChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("❌ LangChain not installed. Run: pip install langchain langchain-openai")
    exit(1)

load_dotenv()

print("🧠 AgentMind - LangChain Integration Example\n")

# 1. Create memory
memory = LangChainMemory(
    agent_id="langchain-agent",
    memory_key="history",
    return_messages=False
)

# 2. Create LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# 3. Create simple conversation chain
from langchain.prompts import PromptTemplate

template = """
You are a helpful assistant with persistent memory.

Previous conversation:
{history}

Current user message: {input}

Your response:
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# 4. Test conversation
print("💬 Testing conversation with memory...\n")

# First question
print("User: What's my name?")
response1 = chain.predict(input="My name is Alex and I'm a Python developer")
print(f"Assistant: {response1}\n")

# Second question (should remember context)
print("User: What do you know about me?")
response2 = chain.predict(input="What do you remember about me?")
print(f"Assistant: {response2}\n")

# Third question (memory check)
print("User: What programming language do I use?")
response3 = chain.predict(input="What programming language do I use?")
print(f"Assistant: {response3}\n")

print("✅ Example completed!")
