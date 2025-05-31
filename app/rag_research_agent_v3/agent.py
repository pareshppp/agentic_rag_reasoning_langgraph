"""RAG Research Agent using Langgraph and Gemini 2.5 Pro."""

import os
from typing import Dict, List, Optional, Any, Callable, TypedDict
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .utils.state import AgentStateV3
from .utils.nodes import (
    initialize_agent_state,
    reasoning_node,
    tool_executor_node,
    generate_final_answer,
    route_after_reasoning,
)
from .utils.tools import create_tools

# Load environment variables from .env file
load_dotenv()

class AgentConfigurationV3(BaseModel):
    """Configuration for the RAG Research Agent V3."""
    
    # MongoDB connection parameters
    mongodb_conn_string: str = Field(default=os.getenv("MONGODB_URI"), description="MongoDB connection string")
    database_name: str = Field(..., description="MongoDB database name")
    collection_name: str = Field(..., description="MongoDB collection name")
    index_name: str = Field(..., description="MongoDB vector index name")
    
    # Model parameters
    llm_model_name: str = Field(default="gemini-2.5-pro", description="LLM model name")
    embedding_model_name: str = Field(default="models/embedding-001", description="Embedding model name")
    llm_temperature_default: float = Field(default=0.0, description="Default temperature for the LLM")
    llm_temperature_final: float = Field(default=0.2, description="Temperature for final answer generation")
    
    # Retriever parameters
    use_multi_query: bool = Field(default=True, description="Whether to use multi-query retriever")
    top_k: int = Field(default=5, description="Number of documents to retrieve")


def create_agent_v3_runnable(
    config: AgentConfigurationV3,
) -> Callable:
    """Create a runnable agent from the configuration."""
    # Create LLM for reasoning
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default,
        convert_system_message_to_human=True,
    )
    
    # Create LLM for final answer generation
    llm_final = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_final,
        convert_system_message_to_human=True,
    )
    
    # Create tools
    tools = create_tools(
        mongodb_conn_string=config.mongodb_conn_string,
        database_name=config.database_name,
        collection_name=config.collection_name,
        index_name=config.index_name,
        embedding_model_name=config.embedding_model_name,
        llm_model_name=config.llm_model_name,
        temperature=config.llm_temperature_default,
        use_multi_query=config.use_multi_query,
        top_k=config.top_k,
    )
    
    # Create the graph
    workflow = StateGraph(AgentStateV3)
    
    # Add nodes to the graph
    workflow.add_node("initialize", lambda state: initialize_agent_state(state, state.user_query))
    workflow.add_node("reasoning", lambda state: reasoning_node(state, llm))
    workflow.add_node("tool_executor", lambda state: tool_executor_node(state, tools))
    workflow.add_node("generate_final_answer", lambda state: generate_final_answer(state, llm_final))
    
    # Add edges to the graph
    workflow.add_edge("initialize", "reasoning")
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "tool_executor": "tool_executor",
            "reasoning": "reasoning",
            "generate_final_answer": "generate_final_answer",
        },
    )
    workflow.add_edge("tool_executor", "reasoning")
    workflow.add_edge("generate_final_answer", END)
    
    # Set the entry point
    workflow.set_entry_point("initialize")
    
    # Compile the graph
    app = workflow.compile()
    
    # Create the runnable
    def agent_runnable(query: str) -> str:
        """Run the agent with a query."""
        state = AgentStateV3(user_query=query)
        result = app.invoke(state)
        return result.final_answer
    
    return agent_runnable


def create_studio_agent(
    mongodb_conn_string: str,
    database_name: str,
    collection_name: str,
    index_name: str,
    llm_model_name: str = "gemini-2.5-pro",
    embedding_model_name: str = "models/embedding-001",
    llm_temperature_default: float = 0.0,
    llm_temperature_final: float = 0.2,
    use_multi_query: bool = True,
    top_k: int = 5,
) -> StateGraph:
    """Create a Langgraph Studio compatible agent."""
    # Create configuration
    config = AgentConfigurationV3(
        mongodb_conn_string=mongodb_conn_string,
        database_name=database_name,
        collection_name=collection_name,
        index_name=index_name,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_temperature_default=llm_temperature_default,
        llm_temperature_final=llm_temperature_final,
        use_multi_query=use_multi_query,
        top_k=top_k,
    )
    
    # Create LLM for reasoning
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default,
        convert_system_message_to_human=True,
    )
    
    # Create LLM for final answer generation
    llm_final = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_final,
        convert_system_message_to_human=True,
    )
    
    # Create tools
    tools = create_tools(
        mongodb_conn_string=config.mongodb_conn_string,
        database_name=config.database_name,
        collection_name=config.collection_name,
        index_name=config.index_name,
        embedding_model_name=config.embedding_model_name,
        llm_model_name=config.llm_model_name,
        temperature=config.llm_temperature_default,
        use_multi_query=config.use_multi_query,
        top_k=config.top_k,
    )
    
    # Create the graph
    workflow = StateGraph(AgentStateV3)
    
    # Add nodes to the graph
    workflow.add_node("initialize", lambda state: initialize_agent_state(state, state.user_query))
    workflow.add_node("reasoning", lambda state: reasoning_node(state, llm))
    workflow.add_node("tool_executor", lambda state: tool_executor_node(state, tools))
    workflow.add_node("generate_final_answer", lambda state: generate_final_answer(state, llm_final))
    
    # Add edges to the graph
    workflow.add_edge("initialize", "reasoning")
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "tool_executor": "tool_executor",
            "reasoning": "reasoning",
            "generate_final_answer": "generate_final_answer",
        },
    )
    workflow.add_edge("tool_executor", "reasoning")
    workflow.add_edge("generate_final_answer", END)
    
    # Set the entry point
    workflow.set_entry_point("initialize")
    
    return workflow
