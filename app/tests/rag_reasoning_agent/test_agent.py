import os
import sys
import json
import pytest
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from my_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_reasoning_agent.utils.state import AgentState
from rag_reasoning_agent.utils.config import AgentConfiguration
from rag_reasoning_agent.agent import agent_runnable
from langgraph.graph import END

load_dotenv()

DUMMY_AGENT_TEST_FILE_PATH = "dummy_agent_test_file.txt"
DUMMY_AGENT_TEST_FILE_CONTENT = "This is content from the agent's FileReadTool test, executed via the agent graph."

@pytest.fixture
def setup_and_teardown_dummy_file():
    with open(DUMMY_AGENT_TEST_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_AGENT_TEST_FILE_CONTENT)
    yield
    if os.path.exists(DUMMY_AGENT_TEST_FILE_PATH):
        os.remove(DUMMY_AGENT_TEST_FILE_PATH)

def create_initial_agent_state_for_test(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query, chat_history=[], query_type="", plan=[],
        current_step_index=0, retrieved_documents=[], file_contents={},
        final_answer="", missing_info_request="", intermediate_steps=[]
    )

GOOGLE_API_KEY_PRESENT = bool(os.getenv("GOOGLE_API_KEY"))

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
def test_greeting_flow():
    initial_state = create_initial_agent_state_for_test(user_query="Hello, how are you?")
    final_state = {}

    for event_output in agent_runnable.stream(initial_state, {"recursion_limit": 10}):
        if END in event_output:
            final_state = event_output[END]
            break
    else:
        pytest.fail("Graph execution did not reach END within recursion limit for greeting flow.")

    assert final_state.get('final_answer') is not None, "Final answer should exist in the final state."
    assert "Hello! I am a RAG agent." in final_state.get('final_answer', ""), "Greeting response not found or incorrect."
    assert final_state.get('query_type') == "greeting", "Query type should be 'greeting'."

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls and tools")
def test_file_read_flow(setup_and_teardown_dummy_file):
    initial_state = create_initial_agent_state_for_test(
        user_query=f"Please read the file '{DUMMY_AGENT_TEST_FILE_PATH}' and tell me its content."
    )
    final_state = {}

    for event_output in agent_runnable.stream(initial_state, {"recursion_limit": 15}):
        if END in event_output:
            final_state = event_output[END]
            break
    else:
        pytest.fail("Graph execution did not reach END within recursion limit for file read flow.")

    assert final_state.get('final_answer') is not None, "Final answer should exist in the final state for file read flow."
    assert DUMMY_AGENT_TEST_FILE_CONTENT in final_state.get('final_answer', ""), \
        f"Expected dummy file content not found in final answer. Answer was: '{final_state.get('final_answer', '')}'"

def test_default_config_values():
    config = AgentConfiguration()
    assert config is not None, "Default config should be instantiated."
    assert config.llm_model_name == "gemini-2.5-flash-preview-05-20"

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
def test_followup_question_flow():
    initial_state = create_initial_agent_state_for_test(
        user_query="What is the capital of France? And what is its population?"
    )
    final_state = {}

    for event_output in agent_runnable.stream(initial_state, {"recursion_limit": 15}):
        if END in event_output:
            final_state = event_output[END]
            break
    else:
        pytest.fail("Graph execution did not reach END within recursion limit for follow-up question flow.")

    assert final_state.get('final_answer') is not None, "Final answer should exist in the final state for follow-up question."
    assert "Paris" in final_state.get('final_answer', ""), "Expected answer about Paris not found."
    assert "population" in final_state.get('final_answer', "").lower(), "Expected answer about population not found."

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
def test_unknown_query_type_flow():
    initial_state = create_initial_agent_state_for_test(
        user_query="Blorptastic quantum flux inversion?"
    )
    final_state = {}

    for event_output in agent_runnable.stream(initial_state, {"recursion_limit": 10}):
        if END in event_output:
            final_state = event_output[END]
            break
    else:
        pytest.fail("Graph execution did not reach END within recursion limit for unknown query type flow.")

    assert final_state.get('final_answer') is not None, "Final answer should exist in the final state for unknown query."
    assert "don't understand" in final_state.get('final_answer', "").lower() or \
           "not sure" in final_state.get('final_answer', "").lower() or \
           "could not" in final_state.get('final_answer', "").lower(), \
           "Agent should indicate it cannot answer the unknown query."

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for agent flows involving LLM calls")
def test_empty_query_flow():
    initial_state = create_initial_agent_state_for_test(user_query="")
    final_state = {}

    for event_output in agent_runnable.stream(initial_state, {"recursion_limit": 5}):
        if END in event_output:
            final_state = event_output[END]
            break
    else:
        pytest.fail("Graph execution did not reach END within recursion limit for empty query flow.")

    assert final_state.get('final_answer') is not None, "Final answer should exist in the final state for empty query."
    assert "please provide" in final_state.get('final_answer', "").lower() or \
           "no question" in final_state.get('final_answer', "").lower(), \
           "Agent should prompt for a valid query when input is empty."
