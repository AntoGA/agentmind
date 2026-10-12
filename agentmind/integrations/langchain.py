"""
LangChain Integration
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from langchain.memory import BaseMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object

from agentmind.core.memory import Memory, MemoryConfig


class LangChainMemory(BaseMemory if LANGCHAIN_AVAILABLE else object):
    """
    LangChain-compatible memory wrapper.
    
    Example:
        >>> from agentmind.integrations import LangChainMemory
        >>> from langchain.agents import initialize_agent
        >>> 
        >>> memory = LangChainMemory(agent_id="my-agent")
        >>> agent = initialize_agent(tools, llm, memory=memory)
        >>> agent.run("What did we discuss?")
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Optional[MemoryConfig] = None,
        memory_key: str = "history",
        return_messages: bool = False,
        **kwargs
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Run: pip install langchain")
        
        super().__init__()
        
        self.agent_id = agent_id
        self.memory = Memory(config=config, **kwargs)
        self.memory_key = memory_key
        self.return_messages = return_messages
    
    @property
    def memory_variables(self) -> List[str]:
        """Memory variables for LangChain"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memories for context.
        
        Args:
            inputs: Input data from user
        
        Returns:
            Dictionary with memory variable
        """
        # Extract query from inputs
        query = inputs.get("input", "")
        
        # Get relevant memories
        memories = self.memory.recall(
            agent_id=self.agent_id,
            query=query,
            limit=10
        )
        
        # Format for LangChain
        if self.return_messages:
            # Return as list of messages
            messages = []
            for mem in memories:
                if mem.metadata.get("role") == "user":
                    messages.append(HumanMessage(content=mem.content))
                else:
                    messages.append(AIMessage(content=mem.content))
            
            return {self.memory_key: messages}
        else:
            # Return as string
            history = "\n".join([f"- {mem.content}" for mem in memories])
            return {self.memory_key: history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context.
        
        Args:
            inputs: Input data (user question)
            outputs: Output data (agent response)
        """
        # Save user question
        user_input = inputs.get("input", "")
        if user_input:
            self.memory.remember(
                agent_id=self.agent_id,
                content=user_input,
                metadata={"role": "user", "type": "conversation"}
            )
        
        # Save agent response
        agent_output = outputs.get("output", "")
        if agent_output:
            self.memory.remember(
                agent_id=self.agent_id,
                content=agent_output,
                metadata={"role": "assistant", "type": "conversation"}
            )
    
    def clear(self) -> None:
        """Clear memory"""
        self.memory.forget(agent_id=self.agent_id)
