"""Node functions for the RAG Research Agent graph."""

from typing import Dict, List, Tuple, Any, Optional, Annotated, TypedDict
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
from .state import AgentStateV3
from .prompts import REASONING_PROMPT, FINAL_ANSWER_PROMPT


def initialize_agent_state(state: AgentStateV3, user_query: str) -> AgentStateV3:
    """Initialize the agent state with the user query."""
    # Add the user query to chat history
    state.chat_history.append(HumanMessage(content=user_query))
    state.user_query = user_query
    return state


def reasoning_node(
    state: AgentStateV3,
    llm: BaseLanguageModel,
) -> AgentStateV3:
    """Reasoning node for the agent."""
    # Prepare the input for the reasoning prompt
    reasoning_input = {
        "chat_history": state.chat_history,
        "intermediate_steps": state.intermediate_steps,
        "user_query": state.user_query,
    }
    
    # Invoke the LLM with the reasoning prompt
    response = llm.invoke(REASONING_PROMPT.format_messages(**reasoning_input))
    
    # Add the response to intermediate steps
    state.intermediate_steps.append(response)
    
    # Extract tool calls if any
    tool_calls = extract_tool_calls(response.content)
    if tool_calls:
        state.tool_calls = tool_calls
        state.use_tools = True
        state.continue_reasoning = True
        state.provide_answer = False
    else:
        # Check if this is a final answer
        state.use_tools = False
        state.continue_reasoning = False
        state.provide_answer = True
        state.final_answer = response.content
    
    return state


def extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Extract tool calls from the LLM response content."""
    tool_calls = []
    
    # Pattern for tool calls in the format: tool_name({"param1": "value1", "param2": "value2"})
    pattern = r'(\w+)\(({.*?})\)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        tool_name, args_str = match
        try:
            args = json.loads(args_str)
            tool_calls.append({
                "name": tool_name,
                "arguments": args,
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract arguments manually
            args_pattern = r'"(\w+)":\s*"([^"]*)"'
            args_matches = re.findall(args_pattern, args_str)
            args = {key: value for key, value in args_matches}
            tool_calls.append({
                "name": tool_name,
                "arguments": args,
            })
    
    return tool_calls


def tool_executor_node(
    state: AgentStateV3,
    tools: List[BaseTool],
) -> AgentStateV3:
    """Execute tools based on the agent's tool calls."""
    tool_dict = {tool.name: tool for tool in tools}
    
    for tool_call in state.tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        
        if tool_name in tool_dict:
            tool = tool_dict[tool_name]
            
            # Execute the tool
            try:
                if isinstance(arguments, dict):
                    # If arguments is a dictionary, pass it as kwargs
                    result = tool.invoke(arguments)
                else:
                    # If arguments is a string, pass it directly
                    result = tool.invoke(arguments)
                
                # Create a tool message with the result
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_name,
                    name=tool_name,
                )
                
                # Add the tool message to intermediate steps
                state.intermediate_steps.append(tool_message)
                
                # Add the tool result to the tool results list
                state.tool_results.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                })
                
            except Exception as e:
                # Handle tool execution errors
                error_message = f"Error executing tool {tool_name}: {str(e)}"
                tool_message = ToolMessage(
                    content=error_message,
                    tool_call_id=tool_name,
                    name=tool_name,
                )
                state.intermediate_steps.append(tool_message)
        else:
            # Handle unknown tool
            error_message = f"Unknown tool: {tool_name}"
            tool_message = ToolMessage(
                content=error_message,
                tool_call_id=tool_name,
                name="unknown_tool",
            )
            state.intermediate_steps.append(tool_message)
    
    # Clear tool calls for the next iteration
    state.tool_calls = []
    
    # Continue reasoning after tool execution
    state.continue_reasoning = True
    state.use_tools = False
    
    return state


def generate_final_answer(
    state: AgentStateV3,
    llm: BaseLanguageModel,
) -> AgentStateV3:
    """Generate the final answer for the user."""
    # Prepare the input for the final answer prompt
    final_answer_input = {
        "chat_history": state.chat_history,
        "intermediate_steps": state.intermediate_steps,
    }
    
    # Invoke the LLM with the final answer prompt
    response = llm.invoke(FINAL_ANSWER_PROMPT.format_messages(**final_answer_input))
    
    # Set the final answer
    state.final_answer = response.content
    
    # Add the final answer to chat history
    state.chat_history.append(AIMessage(content=state.final_answer))
    
    # Reset state for next interaction
    state.continue_reasoning = False
    state.use_tools = False
    state.provide_answer = False
    state.intermediate_steps = []
    state.tool_calls = []
    state.tool_results = []
    
    return state


def route_after_reasoning(state: AgentStateV3) -> str:
    """Route to the next node after reasoning."""
    if state.provide_answer:
        return "generate_final_answer"
    elif state.use_tools:
        return "tool_executor"
    elif state.continue_reasoning:
        return "reasoning"
    else:
        return "generate_final_answer"
