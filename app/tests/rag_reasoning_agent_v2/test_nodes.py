import pytest
from unittest.mock import MagicMock, patch
import typing

from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage, ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI

# Attempt to import from the V2 agent's utils
try:
    from app.rag_reasoning_agent_v2.utils.state import AgentStateV2
    from app.rag_reasoning_agent_v2.utils.config import AgentConfigurationV2
    from app.rag_reasoning_agent_v2.utils.nodes import (
        reasoning_node,
        tool_executor_node_v2,
        route_after_reasoning
    )
except ImportError:
    print("Attempting fallback imports for test_nodes.py - ensure PYTHONPATH is correct")
    from rag_reasoning_agent_v2.utils.state import AgentStateV2
    from rag_reasoning_agent_v2.utils.config import AgentConfigurationV2
    from rag_reasoning_agent_v2.utils.nodes import (
        reasoning_node,
        tool_executor_node_v2,
        route_after_reasoning
    )

# --- Fixtures --- 

@pytest.fixture
def default_v2_config() -> AgentConfigurationV2:
    """Provides a default AgentConfigurationV2 instance for tests."""
    return AgentConfigurationV2()

@pytest.fixture
def mock_llm_with_tools() -> MagicMock:
    """Provides a MagicMock for ChatGoogleGenerativeAI (tool-bound)."""
    mock = MagicMock(spec=ChatGoogleGenerativeAI)
    # Configure default return values if needed for specific tests later
    return mock

@pytest.fixture
def mock_tool() -> MagicMock:
    """Provides a MagicMock for a generic LangChain Tool."""
    tool = MagicMock(spec=Tool)
    tool.name = "MockTool"
    tool.description = "A mock tool for testing."
    # tool.invoke.return_value = "Mock tool output"
    return tool

@pytest.fixture
def initial_agent_state() -> AgentStateV2:
    """Provides a basic initial AgentStateV2 for tests."""
    return AgentStateV2(
        user_query="Test query",
        chat_history=[
            SystemMessage(content="System prompt"),
            HumanMessage(content="Test query")
        ],
        intermediate_steps=[],
        retrieved_documents=[],
        file_contents={},
        final_answer=None
    )

# --- Tests for reasoning_node --- 

def test_reasoning_node_direct_answer(initial_agent_state: AgentStateV2, default_v2_config: AgentConfigurationV2, mock_llm_with_tools: MagicMock):
    """Test reasoning_node when LLM decides to answer directly."""
    # Configure mock_llm_with_tools to return an AIMessage without tool calls
    ai_response_content = "This is a direct answer."
    mock_llm_with_tools.invoke.return_value = AIMessage(content=ai_response_content)

    # reasoning_node returns a partial state update
    partial_updated_state = reasoning_node(initial_agent_state, default_v2_config, mock_llm_with_tools)

    mock_llm_with_tools.invoke.assert_called_once()
    # Check that intermediate_steps in the partial update contains the AI's direct answer
    assert 'intermediate_steps' in partial_updated_state
    assert len(partial_updated_state['intermediate_steps']) == 1
    last_intermediate_message = partial_updated_state['intermediate_steps'][-1]
    assert isinstance(last_intermediate_message, AIMessage)
    assert last_intermediate_message.content == ai_response_content
    assert not last_intermediate_message.tool_calls
    assert partial_updated_state.get('final_answer') == ai_response_content # Reasoning node sets final_answer for direct answer


def test_reasoning_node_tool_call(initial_agent_state: AgentStateV2, default_v2_config: AgentConfigurationV2, mock_llm_with_tools: MagicMock):
    """Test reasoning_node when LLM decides to call a tool."""
    tool_call_id = "tool_call_123"
    tool_name = "MockTool"
    tool_args = {"arg1": "value1"}

    # Configure mock_llm_with_tools to return an AIMessage with a tool call
    ai_response_with_tool_call = AIMessage(
        content="I need to use a tool.",
        tool_calls=[ToolCall(id=tool_call_id, name=tool_name, args=tool_args)] # Use ToolCall model
    )
    mock_llm_with_tools.invoke.return_value = ai_response_with_tool_call

    partial_updated_state = reasoning_node(initial_agent_state, default_v2_config, mock_llm_with_tools)

    mock_llm_with_tools.invoke.assert_called_once()
    # Check that intermediate_steps in the partial update contains the AI's message with tool call
    assert 'intermediate_steps' in partial_updated_state
    assert len(partial_updated_state['intermediate_steps']) == 1
    last_intermediate_message = partial_updated_state['intermediate_steps'][-1]
    assert isinstance(last_intermediate_message, AIMessage)
    assert last_intermediate_message.content == "I need to use a tool."
    assert len(last_intermediate_message.tool_calls) == 1
    assert last_intermediate_message.tool_calls[0]['id'] == tool_call_id
    assert last_intermediate_message.tool_calls[0]['name'] == tool_name
    assert last_intermediate_message.tool_calls[0]['args'] == tool_args
    assert partial_updated_state.get('final_answer') is None


# --- Tests for tool_executor_node_v2 --- 

def test_tool_executor_node_v2_single_tool(initial_agent_state: AgentStateV2, mock_tool: MagicMock):
    """Test tool_executor_node_v2 with a single successful tool call."""
    tool_call_id = "tool_call_abc"
    tool_name = mock_tool.name
    tool_args = {"input": "test input"}
    tool_output = "Mock tool output for test"
    mock_tool.invoke.return_value = tool_output

    # Prepare state with a tool call from reasoning_node in intermediate_steps
    # The tool_executor_node_v2 expects the AIMessage with tool_calls in intermediate_steps
    initial_agent_state['intermediate_steps'] = [
        AIMessage(
            content="Using a tool",
            tool_calls=[ToolCall(id=tool_call_id, name=tool_name, args=tool_args)]
        )
    ]
    tools_map = {tool_name: mock_tool}

    updated_partial_state = tool_executor_node_v2(initial_agent_state, tools_map)

    mock_tool.invoke.assert_called_once_with(tool_args)
    assert 'intermediate_steps' in updated_partial_state
    assert len(updated_partial_state['intermediate_steps']) == 2 # AIMessage + ToolMessage
    tool_message = updated_partial_state['intermediate_steps'][-1] # ToolMessage is the last one
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == tool_output
    assert tool_message.tool_call_id == tool_call_id
    assert updated_partial_state.get('final_answer') is None # Successful execution shouldn't set final_answer here


def test_tool_executor_node_v2_tool_error(initial_agent_state: AgentStateV2, mock_tool: MagicMock):
    """Test tool_executor_node_v2 when a tool raises an exception."""
    tool_call_id = "tool_call_err"
    tool_name = mock_tool.name
    tool_args = {"input": "bad input"}
    error_message = "Tool failed spectacularly!"
    mock_tool.invoke.side_effect = Exception(error_message)

    initial_agent_state['intermediate_steps'] = [
        AIMessage(
            content="Using a tool that will fail",
            tool_calls=[ToolCall(id=tool_call_id, name=tool_name, args=tool_args)]
        )
    ]
    tools_map = {tool_name: mock_tool}

    updated_partial_state = tool_executor_node_v2(initial_agent_state, tools_map)

    mock_tool.invoke.assert_called_once_with(tool_args)
    assert 'intermediate_steps' in updated_partial_state
    assert len(updated_partial_state['intermediate_steps']) == 2 # AIMessage + ToolMessage
    tool_message = updated_partial_state['intermediate_steps'][-1] # ToolMessage is the last one
    assert isinstance(tool_message, ToolMessage)
    assert f"Error executing tool {tool_name}: {error_message}" in tool_message.content
    assert tool_message.tool_call_id == tool_call_id
    assert updated_partial_state.get('final_answer') is None


def test_tool_executor_node_v2_no_tool_calls(initial_agent_state: AgentStateV2):
    """Test tool_executor_node_v2 when there are no tool calls in the last AI message."""
    # Last message is a direct answer, no tool calls
    initial_agent_state['intermediate_steps'] = [AIMessage(content="Direct answer, no tools.")]
    tools_map = {}
    
    updated_partial_state = tool_executor_node_v2(initial_agent_state, tools_map)
    
    assert 'intermediate_steps' in updated_partial_state
    assert len(updated_partial_state['intermediate_steps']) == 1 # No tool messages added
    # State should remain largely unchanged by tool_executor if no tools to execute
    assert updated_partial_state['intermediate_steps'][0].content == "Direct answer, no tools."


# --- Tests for route_after_reasoning --- 

def test_route_after_reasoning_to_tool_executor(initial_agent_state: AgentStateV2):
    """Test route_after_reasoning routes to 'tool_executor' if tool calls exist."""
    # Add an AIMessage with tool calls to intermediate_steps
    initial_agent_state['intermediate_steps'] = [
        AIMessage(content="Need tool", tool_calls=[ToolCall(id="1", name="TestTool", args={})])
    ]
    initial_agent_state['final_answer'] = None # Ensure final_answer is not set

    route = route_after_reasoning(initial_agent_state)
    assert route == "tool_executor"


def test_route_after_reasoning_to_end_with_final_answer(initial_agent_state: AgentStateV2):
    """Test route_after_reasoning routes to END if final_answer is set."""
    initial_agent_state['final_answer'] = "This is the final answer."
    # Even if there were tool calls in intermediate_steps, final_answer takes precedence
    initial_agent_state['intermediate_steps'] = [
        AIMessage(content="Previous tool use", tool_calls=[ToolCall(id="1", name="TestTool", args={})])
    ]

    route = route_after_reasoning(initial_agent_state)
    assert route == "__end__"


def test_route_after_reasoning_to_end_no_tool_calls_no_final_answer_but_last_ai_msg(initial_agent_state: AgentStateV2):
    """Test route_after_reasoning routes to END if last AI message in intermediate_steps has no tool calls and no final_answer is explicitly set."""
    initial_agent_state['intermediate_steps'] = [AIMessage(content="This is a direct AI response.")] # No tool calls
    initial_agent_state['final_answer'] = None

    route = route_after_reasoning(initial_agent_state)
    assert route == "__end__"


def test_route_after_reasoning_empty_history_or_no_ai_message(initial_agent_state: AgentStateV2):
    """Test route_after_reasoning behavior with empty intermediate_steps (should default to END)."""
    initial_agent_state['intermediate_steps'] = []
    initial_agent_state['final_answer'] = None
    route = route_after_reasoning(initial_agent_state)
    assert route == "__end__"


def test_route_after_reasoning_intermediate_steps_not_ai_message(initial_agent_state: AgentStateV2):
    """Test route_after_reasoning routes to END if last message in intermediate_steps is not an AIMessage."""
    initial_agent_state['intermediate_steps'] = [ToolMessage(content="tool output", tool_call_id="t1")]
    initial_agent_state['final_answer'] = None
    route = route_after_reasoning(initial_agent_state)
    assert route == "__end__"
