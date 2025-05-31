"""State definition for the RAG Research Agent."""

from typing import Dict, List, Optional, Any, Union
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from pydantic import BaseModel, Field


class AgentStateV3(BaseModel):
    """State for the RAG Research Agent V3."""

    # Chat history between the user and the agent
    chat_history: List[BaseMessage] = Field(default_factory=list)
    
    # Intermediate steps for the agent's reasoning
    intermediate_steps: List[BaseMessage] = Field(default_factory=list)
    
    # Final answer to return to the user
    final_answer: Optional[str] = None
    
    # Tool calls to be executed
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Tool results from executed tools
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Flag to indicate if the agent should continue reasoning
    continue_reasoning: bool = True
    
    # Flag to indicate if the agent should use tools
    use_tools: bool = False
    
    # Flag to indicate if the agent should provide a final answer
    provide_answer: bool = False
    
    # User query
    user_query: Optional[str] = None
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to the state."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of state attributes."""
        setattr(self, key, value)
