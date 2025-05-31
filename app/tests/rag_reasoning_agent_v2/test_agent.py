import os
import sys
import pytest
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.rag_reasoning_agent_v2.utils.state import AgentStateV2
from app.rag_reasoning_agent_v2.utils.config import AgentConfigurationV2
from app.rag_reasoning_agent_v2.agent import agent_v2_runnable # Assuming this is the V2 agent graph

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, ToolCall
from langgraph.graph import END
from unittest.mock import patch # Added patch

# Attempt to import V2 specific modules for testing context
# This mirrors the try-except block in agent.py for robustness
try:
    from app.rag_reasoning_agent_v2.agent import create_agent_v2_runnable, AgentConfigurationV2, AgentStateV2, END
    from app.rag_reasoning_agent_v2.utils.tools import DUMMY_DOCS # For retriever test
    from app.rag_reasoning_agent_v2.utils.state import AgentStateV2 # Explicit import if needed
except ImportError:
    print("Error importing V2 agent components for test_agent.py. Ensure PYTHONPATH is correct or running via pytest from root.")
    # Fallbacks if running in a different context, adjust as necessary or ensure test environment is set up
    from utils.tools import DUMMY_DOCS # Assuming utils is accessible
    # Add other fallbacks if create_agent_v2_runnable etc. are not found via app path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DUMMY_AGENT_TEST_FILE_PATH = "dummy_agent_v2_test_file.txt"
DUMMY_AGENT_TEST_FILE_CONTENT = "This is content from the agent's FileReadTool test for V2, executed via the agent graph."

@pytest.fixture
def setup_and_teardown_dummy_file():
    with open(DUMMY_AGENT_TEST_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_AGENT_TEST_FILE_CONTENT)
    yield
    if os.path.exists(DUMMY_AGENT_TEST_FILE_PATH):
        os.remove(DUMMY_AGENT_TEST_FILE_PATH)

def create_initial_agent_state_v2_for_test(user_query: str) -> AgentStateV2:
    # In a real scenario, a SystemMessage might be part of the initial chat_history
    # For focused testing of a single turn, starting with HumanMessage is often simpler.
    return AgentStateV2(
        chat_history=[HumanMessage(content=user_query)],
        intermediate_steps=[],
        file_contents={},
        final_answer=None
    )

GOOGLE_API_KEY_PRESENT = bool(os.getenv("GOOGLE_API_KEY"))

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
def test_direct_answer_flow_v2():
    """Test a simple query that should result in a direct answer without tool usage."""
    initial_state = create_initial_agent_state_v2_for_test(user_query="Hello, how are you today?")
    final_state = agent_v2_runnable.invoke(initial_state, {"recursion_limit": 5})
    
    print(f"DEBUG test_direct_answer_flow_v2: Final state from invoke: {final_state}")

    assert final_state is not None, "Final state from invoke should not be None."
    # For immediate END routes, invoke() returns the state of the last node, not {END: state}
    # assert END in final_state, f"END key not found in final_state from invoke. State: {final_state}"
    
    actual_final_agent_state = final_state # The returned state is the final agent state

    assert actual_final_agent_state is not None, "Actual final agent state should not be None after graph execution."
    assert actual_final_agent_state.get('final_answer') is not None, "Final answer should exist in the final agent state."
    # For a direct answer, the LLM might return an empty string if it has nothing specific to say beyond a greeting
    # or if it's configured to be very terse. The main thing is that it didn't try to use tools.
    # assert "I'm doing well" in actual_final_agent_state.get('final_answer', ""), \
    #     f"Expected greeting not found. Answer: '{actual_final_agent_state.get('final_answer', '')}'"
    # Ensure no tool calls were made if this is a direct answer test
    if actual_final_agent_state.get('intermediate_steps'):
        for step in actual_final_agent_state['intermediate_steps']:
            if isinstance(step, AIMessage):
                assert not step.tool_calls, f"Expected no tool calls for direct answer, but found: {step.tool_calls}"

@pytest.fixture
def setup_faiss_vector_store():
    """Fixture to create an in-memory FAISS vector store with DUMMY_DOCS."""
    if not GOOGLE_API_KEY_PRESENT:
        pytest.skip("GOOGLE_API_KEY required for embeddings for FAISS vector store.")
    try:
        # Using a known model for embeddings, consistent with agent if possible
        # The specific model name might come from AgentConfigurationV2 defaults or be hardcoded for test stability
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Or config.embedding_model_name
        vector_store = FAISS.from_documents(DUMMY_DOCS, embeddings)
        print("DEBUG: In-memory FAISS vector store created for test with DUMMY_DOCS.")
        return vector_store
    except Exception as e:
        pytest.fail(f"Failed to create FAISS vector store for test: {e}")

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls and tools")
@patch('app.rag_reasoning_agent_v2.agent.ChatGoogleGenerativeAI.invoke') # Patch the LLM invoke
def test_file_read_tool_flow_v2(mock_llm_invoke, setup_and_teardown_dummy_file):
    """Test a query that requires reading a file using FileReadTool, with mocked LLM responses."""

    # 1. Mock LLM to return a tool call for FileReadTool
    mock_tool_call_id = "tool_call_file_read_123"
    ai_message_with_tool_call = AIMessage(
        content="", 
        tool_calls=[
            ToolCall(
                name="FileReadTool", 
                args={"file_path": DUMMY_AGENT_TEST_FILE_PATH}, 
                id=mock_tool_call_id
            )
        ]
    )

    # 2. Mock LLM to return a final answer after tool execution
    # The LLM would typically summarize or use the tool's output.
    expected_final_answer_content = f"The content of the file '{DUMMY_AGENT_TEST_FILE_PATH}' is: {DUMMY_AGENT_TEST_FILE_CONTENT}"
    ai_message_with_final_answer = AIMessage(content=expected_final_answer_content)

    mock_llm_invoke.side_effect = [ai_message_with_tool_call, ai_message_with_final_answer]

    user_query = f"Please read the file '{DUMMY_AGENT_TEST_FILE_PATH}' and tell me its content."
    initial_state = create_initial_agent_state_v2_for_test(user_query=user_query)
    final_state_from_invoke = agent_v2_runnable.invoke(initial_state, {"recursion_limit": 10})

    print(f"DEBUG test_file_read_tool_flow_v2: Final state from invoke: {final_state_from_invoke}")

    assert final_state_from_invoke is not None, "Final state from invoke should not be None for file read."
    # For immediate END routes, invoke() returns the state of the last node, not {END: state}
    # assert END in final_state_from_invoke, f"END key not found in final_state_from_invoke for file read. State: {final_state_from_invoke}"

    actual_final_agent_state = final_state_from_invoke # The returned state is the final agent state

    assert actual_final_agent_state is not None, "Actual final agent state should not be None after graph execution for file read."
    assert actual_final_agent_state.get('final_answer') is not None, "Final answer should exist in final agent state for file read flow."
    
    # --- Assertions based on mocked LLM and actual tool execution ---
    intermediate_steps = actual_final_agent_state.get('intermediate_steps', [])

    # Check 1: First LLM call resulted in a tool call
    assert len(intermediate_steps) > 0, "Intermediate steps should not be empty."
    first_step = intermediate_steps[0]
    assert isinstance(first_step, AIMessage), "First step should be an AIMessage from the LLM."
    assert first_step.tool_calls is not None and len(first_step.tool_calls) == 1, "AIMessage should have one tool call."
    tool_call_emitted = first_step.tool_calls[0]
    assert tool_call_emitted['name'] == "FileReadTool", "Tool call should be for FileReadTool."
    assert tool_call_emitted['args'].get("file_path") == DUMMY_AGENT_TEST_FILE_PATH, "Tool call arg mismatch."
    assert tool_call_emitted['id'] == mock_tool_call_id, "Tool call ID mismatch."

    # Check 2: FileReadTool was executed and its output is present
    assert len(intermediate_steps) > 1, "Should be at least two steps (AIMessage tool_call + ToolMessage result)."
    second_step = intermediate_steps[1]
    assert isinstance(second_step, ToolMessage), "Second step should be a ToolMessage from FileReadTool."
    assert second_step.name == "FileReadTool", "ToolMessage name should be FileReadTool."
    assert DUMMY_AGENT_TEST_FILE_CONTENT in second_step.content, "ToolMessage content mismatch."
    assert second_step.tool_call_id == mock_tool_call_id, "ToolMessage tool_call_id should match AIMessage tool_call id."

    # Check 3: Final answer from the second LLM call (which processed tool output)
    assert actual_final_agent_state.get('final_answer') == expected_final_answer_content, \
        f"Final answer mismatch. Expected: '{expected_final_answer_content}'. Got: '{actual_final_agent_state.get('final_answer')}'"

    # Cleanup the dummy file (already handled by fixture, but good to be explicit if not using fixture for cleanup)
    # os.remove(DUMMY_AGENT_TEST_FILE_PATH)

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls and tools")
@patch('app.rag_reasoning_agent_v2.agent.ChatGoogleGenerativeAI.invoke')
def test_retriever_tool_flow_v2(mock_llm_invoke, setup_faiss_vector_store):
    """Test a query that requires using the RetrieverTool, with mocked LLM and in-memory FAISS."""
    vector_store = setup_faiss_vector_store
    test_config = AgentConfigurationV2() # Uses default faiss_index_path, but it won't be used by tool

    # Create agent runnable, passing the pre-initialized vector store
    # This ensures the retriever tool uses our in-memory DUMMY_DOCS store
    agent_under_test = create_agent_v2_runnable(config=test_config, vector_store=vector_store)

    # 1. Mock LLM to return a tool call for RetrieverTool
    mock_retriever_tool_call_id = "tool_call_retriever_456"
    retriever_query = "What is LangGraph?"
    ai_message_with_retriever_call = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="RetrieverTool", 
                args={"query": retriever_query}, 
                id=mock_retriever_tool_call_id
            )
        ]
    )

    # 2. Mock LLM to return a final answer after tool execution (simulating summarization of retrieved docs)
    # For this test, we'll assume the retriever found something about LangGraph from DUMMY_DOCS
    # One of the DUMMY_DOCS is: Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", metadata={'source': 'doc_1'})    
    expected_retrieved_content_summary = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
    ai_message_with_final_answer_from_retrieval = AIMessage(content=f"Based on the retrieved documents: {expected_retrieved_content_summary}")

    # Mock response for MultiQueryRetriever's internal LLM call for query generation
    mock_query_generation_response = AIMessage(content=f"['{retriever_query}']")

    mock_llm_invoke.side_effect = [
        ai_message_with_retriever_call, 
        mock_query_generation_response, 
        ai_message_with_final_answer_from_retrieval
    ]
    
    user_query = "Tell me about LangGraph using the retriever."
    initial_state = create_initial_agent_state_v2_for_test(user_query=user_query)
    
    final_state_from_invoke = agent_under_test.invoke(initial_state, {"recursion_limit": 10})
    print(f"DEBUG test_retriever_tool_flow_v2: Final state from invoke: {final_state_from_invoke}")
    
    actual_final_agent_state = final_state_from_invoke

    assert actual_final_agent_state.get('final_answer') is not None, "Final answer should exist."

    # --- Assertions ---
    intermediate_steps = actual_final_agent_state.get('intermediate_steps', [])

    # Check 1: First LLM call resulted in a RetrieverTool call
    assert len(intermediate_steps) > 0, "Intermediate steps should not be empty."
    first_step_ai_call = intermediate_steps[0]
    assert isinstance(first_step_ai_call, AIMessage)
    assert first_step_ai_call.tool_calls is not None and len(first_step_ai_call.tool_calls) == 1
    tool_call_emitted = first_step_ai_call.tool_calls[0]
    assert tool_call_emitted['name'] == "RetrieverTool"
    assert tool_call_emitted['args'].get("query") == retriever_query
    assert tool_call_emitted['id'] == mock_retriever_tool_call_id

    # Check 2: RetrieverTool was executed and its output is present
    # The actual tool will run against the in-memory FAISS store. Its output will be a string of concatenated docs.
    assert len(intermediate_steps) > 1, "Should be at least AIMessage + ToolMessage."
    second_step_tool_result = intermediate_steps[1]
    assert isinstance(second_step_tool_result, ToolMessage)
    assert second_step_tool_result.name == "RetrieverTool"
    # Check if some part of DUMMY_DOCS content is in the tool's output
    # For example, check for the content of the LangGraph document
    assert "LangGraph is a library for building stateful" in second_step_tool_result.content
    assert second_step_tool_result.tool_call_id == mock_retriever_tool_call_id

    # Check 3: Final answer from the second LLM call (which processed tool output)
    assert actual_final_agent_state.get('final_answer') == ai_message_with_final_answer_from_retrieval.content, \
        f"Final answer mismatch. Expected: '{ai_message_with_final_answer_from_retrieval.content}'. Got: '{actual_final_agent_state.get('final_answer')}'"

    assert mock_llm_invoke.call_count == 3, f"Expected LLM to be invoked 3 times, but was {mock_llm_invoke.call_count}"

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
@patch('app.rag_reasoning_agent_v2.agent.ChatGoogleGenerativeAI.invoke')
def test_unknown_query_handling_v2(mock_llm_invoke):
    """Test a query that the LLM should answer with 'I don't know' without tool usage."""
    test_config = AgentConfigurationV2()
    agent_under_test = create_agent_v2_runnable(config=test_config, vector_store=None) # No vector store needed

    # Mock LLM to return a direct "I don't know" style answer
    unknown_answer_content = "I'm sorry, I don't have information about the color of the sky on Mars during a solar eclipse."
    ai_message_unknown_answer = AIMessage(content=unknown_answer_content)
    mock_llm_invoke.return_value = ai_message_unknown_answer # Only one LLM call expected

    user_query = "What is the color of the sky on Mars during a solar eclipse?"
    initial_state = create_initial_agent_state_v2_for_test(user_query=user_query)

    final_state_from_invoke = agent_under_test.invoke(initial_state, {"recursion_limit": 5})
    print(f"DEBUG test_unknown_query_handling_v2: Final state from invoke: {final_state_from_invoke}")

    actual_final_agent_state = final_state_from_invoke

    assert actual_final_agent_state.get('final_answer') == unknown_answer_content, \
        f"Final answer mismatch. Expected: '{unknown_answer_content}'. Got: '{actual_final_agent_state.get('final_answer')}'"

    # Ensure no tool calls were attempted
    intermediate_steps = actual_final_agent_state.get('intermediate_steps', [])
    # Check if any AIMessage in intermediate_steps has tool_calls
    tool_calls_made = any(
        isinstance(step, AIMessage) and step.tool_calls for step in intermediate_steps
    )
    assert not tool_calls_made, "Expected no tool calls for an unknown query, but tool calls were made."

    # Check that the LLM was called exactly once
    assert mock_llm_invoke.call_count == 1, f"Expected LLM to be invoked once, but was {mock_llm_invoke.call_count}"

@patch('app.rag_reasoning_agent_v2.agent.ChatGoogleGenerativeAI')
def test_default_configuration_v2(MockChatGoogleGenerativeAI):
    """Test that the agent runnable is created with default config values."""
    # 1. Get default configuration values
    default_config = AgentConfigurationV2()

    # 2. Create the agent runnable with the default config
    # We don't need to mock LLM invoke here, just its instantiation.
    # No actual agent execution, just creation.
    # Pass vector_store=None as it's not relevant for this config check.
    try:
        _ = create_agent_v2_runnable(config=default_config, vector_store=None)
    except Exception as e:
        pytest.fail(f"create_agent_v2_runnable failed during default config test: {e}")

    # 3. Assert that ChatGoogleGenerativeAI was called with default values
    assert MockChatGoogleGenerativeAI.called, "ChatGoogleGenerativeAI should have been instantiated."
    
    # Get the arguments passed to ChatGoogleGenerativeAI constructor
    # mock_calls[0] is call(), mock_calls[0][1] is args, mock_calls[0][2] is kwargs
    # However, it's usually easier to use call_args or last_call if only one call is expected.
    # call_args gives a Call object: call_args.args, call_args.kwargs
    
    # Ensure it was called at least once
    MockChatGoogleGenerativeAI.assert_called()
    
    # Get the keyword arguments from the last call (should be the only call)
    called_kwargs = MockChatGoogleGenerativeAI.call_args.kwargs

    assert called_kwargs.get('model') == default_config.llm_model_name, \
        f"LLM model mismatch. Expected: '{default_config.llm_model_name}'. Got: '{called_kwargs.get('model')}'"
    
    assert called_kwargs.get('temperature') == default_config.llm_temperature_default, \
        f"LLM temperature mismatch. Expected: '{default_config.llm_temperature_default}'. Got: '{called_kwargs.get('temperature')}'"
