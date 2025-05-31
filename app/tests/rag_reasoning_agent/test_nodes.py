import os
import sys
import pytest 
import json 
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage 
from langchain_core.documents import Document
from rag_reasoning_agent.utils.state import AgentState
from rag_reasoning_agent.utils.config import AgentConfiguration 
from rag_reasoning_agent.utils.nodes import (
    analyze_query_node,
    planner_node,
    tool_executor_node,
    response_synthesizer_node
)
from rag_reasoning_agent.utils.tools import create_faiss_retriever_tool, create_file_read_tool

load_dotenv()

GOOGLE_API_KEY_PRESENT = bool(os.getenv("GOOGLE_API_KEY"))

def create_initial_agent_state(user_query: str, chat_history: list[BaseMessage] = None, 
                               query_type: str = "", plan: list[str] = None,
                               current_step_index: int = 0, retrieved_documents: list[Document] = None,
                               file_contents: dict = None, final_answer: str = "",
                               missing_info_request: str = "", intermediate_steps: list = None) -> AgentState:
    return AgentState(
        user_query=user_query,
        chat_history=chat_history if chat_history is not None else [],
        query_type=query_type,
        plan=plan if plan is not None else [],
        current_step_index=current_step_index,
        retrieved_documents=retrieved_documents if retrieved_documents is not None else [],
        file_contents=file_contents if file_contents is not None else {},
        final_answer=final_answer,
        missing_info_request=missing_info_request,
        intermediate_steps=intermediate_steps if intermediate_steps is not None else []
    )

DUMMY_EXECUTOR_FILE_PATH = "dummy_executor_test_file.txt"
DUMMY_EXECUTOR_CONTENT = "This is content from the executor test dummy file for nodes test."

@pytest.fixture
def dummy_executor_file():
    with open(DUMMY_EXECUTOR_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_EXECUTOR_CONTENT)
    yield DUMMY_EXECUTOR_FILE_PATH
    if os.path.exists(DUMMY_EXECUTOR_FILE_PATH):
        os.remove(DUMMY_EXECUTOR_FILE_PATH)

@pytest.fixture(scope="module")
def node_test_setup():
    if not GOOGLE_API_KEY_PRESENT:
        print("WARNING: GOOGLE_API_KEY not found. Skipping API-dependent node tests.")
    
    default_config = AgentConfiguration() 
    test_tools_list = [
        create_faiss_retriever_tool(default_config),
        create_file_read_tool(default_config)
    ]
    return {"default_config": default_config, "test_tools_list": test_tools_list}

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for analyze_query_node")
def test_analyze_query_greeting(node_test_setup):
    state = create_initial_agent_state(user_query="Hello there!")
    updated_state = analyze_query_node(node_test_setup["default_config"], state)
    assert updated_state['query_type'] == 'greeting', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for analyze_query_node")
def test_analyze_query_file_read(node_test_setup):
    state = create_initial_agent_state(user_query="Can you read the file named report.txt and also check /tmp/data.csv?")
    updated_state = analyze_query_node(node_test_setup["default_config"], state)
    assert updated_state['query_type'] == 'file_read',  f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"
    assert "report.txt" in updated_state['file_contents']
    assert "/tmp/data.csv" in updated_state['file_contents']

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for analyze_query_node")
def test_analyze_query_ambiguous(node_test_setup):
    history = [AIMessage(content="I can help with information retrieval and file processing.")]
    state = create_initial_agent_state(user_query="Tell me more.", chat_history=history)
    updated_state = analyze_query_node(node_test_setup["default_config"], state)
    assert updated_state['query_type'] == 'ambiguous', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"
    assert len(updated_state['missing_info_request']) > 0

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for analyze_query_node")
def test_analyze_query_retrieval(node_test_setup):
    state = create_initial_agent_state(user_query="What is the capital of France?")
    updated_state = analyze_query_node(node_test_setup["default_config"], state)
    assert updated_state['query_type'] == 'retrieval', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for analyze_query_node")
def test_analyze_query_info_response(node_test_setup):
    history = [
        HumanMessage(content="Can you check the config?"),
        AIMessage(content="Which configuration file are you referring to?")
    ]
    state = create_initial_agent_state(
        user_query="Yes, I meant the main configuration file for the web server.",
        chat_history=history,
        missing_info_request="Which configuration file are you referring to?"
    )
    updated_state = analyze_query_node(node_test_setup["default_config"], state)
    assert updated_state['query_type'] == 'info_response', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for planner_node with LLM")
def test_planner_greeting(node_test_setup): 
    state = create_initial_agent_state(user_query="Hello", query_type="greeting")
    updated_state = planner_node(node_test_setup["default_config"], state)
    assert len(updated_state['plan']) == 0

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for planner_node with LLM")
def test_planner_file_read(node_test_setup):
    state = create_initial_agent_state(
        user_query="Read report.txt", 
        query_type="file_read",
        file_contents={"report.txt": ""} 
    )
    updated_state = planner_node(node_test_setup["default_config"], state)
    assert len(updated_state['plan']) > 0, f"Plan was empty. Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"
    if updated_state['plan']: 
        assert "Read file 'report.txt'" in updated_state['plan'][0]
        assert "Synthesize answer" in updated_state['plan'][-1]

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for planner_node with LLM")
def test_planner_retrieval(node_test_setup):
    state = create_initial_agent_state(user_query="What is LangGraph?", query_type="retrieval")
    updated_state = planner_node(node_test_setup["default_config"], state)
    assert len(updated_state['plan']) > 0, f"Plan was empty. Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}"
    if updated_state['plan']:
        assert "Use retriever for" in updated_state['plan'][0]
        assert "Synthesize answer" in updated_state['plan'][-1]
            
def test_tool_executor_no_plan(node_test_setup):
    state = create_initial_agent_state(user_query="N/A", plan=[])
    updated_state = tool_executor_node(node_test_setup["default_config"], state, tools=node_test_setup["test_tools_list"])
    assert updated_state['current_step_index'] == 0

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for FAISS tool in tool_executor_node")
def test_tool_executor_faiss_retriever(node_test_setup):
    state = create_initial_agent_state(
        user_query="What is LangGraph?", 
        query_type="retrieval",
        plan=["Use retriever for 'LangGraph library features'", "Synthesize answer using all gathered information."]
    )
    updated_state = tool_executor_node(node_test_setup["default_config"], state, tools=node_test_setup["test_tools_list"])
    assert len(updated_state['retrieved_documents']) > 0
    assert isinstance(updated_state['retrieved_documents'][0], Document)
    assert updated_state['current_step_index'] == 1

def test_tool_executor_file_read_success(node_test_setup, dummy_executor_file):
    state = create_initial_agent_state(
        user_query=f"Read {dummy_executor_file}",
        query_type="file_read",
        plan=[f"Read file '{dummy_executor_file}'", "Synthesize answer using all gathered information."],
        file_contents={dummy_executor_file: ""} 
    )
    updated_state = tool_executor_node(node_test_setup["default_config"], state, tools=node_test_setup["test_tools_list"])
    assert updated_state['file_contents'][dummy_executor_file] == DUMMY_EXECUTOR_CONTENT
    assert updated_state['current_step_index'] == 1

def test_tool_executor_file_read_not_found(node_test_setup):
    non_existent_file = "non_existent_for_executor.txt"
    state = create_initial_agent_state(
        user_query=f"Read {non_existent_file}",
        query_type="file_read",
        plan=[f"Read file '{non_existent_file}'", "Synthesize answer using all gathered information."],
        file_contents={non_existent_file: ""}
    )
    updated_state = tool_executor_node(node_test_setup["default_config"], state, tools=node_test_setup["test_tools_list"])
    assert "Error: File not found" in updated_state['file_contents'][non_existent_file]
    assert updated_state['current_step_index'] == 1

def test_tool_executor_synthesize_step(node_test_setup):
    state = create_initial_agent_state(
        user_query="N/A",
        plan=["Synthesize answer using all gathered information."],
        current_step_index=0, 
        retrieved_documents=[Document(page_content="Info A")],
        file_contents={"doc.txt": "Content B"}
    )
    updated_state = tool_executor_node(node_test_setup["default_config"], state, tools=node_test_setup["test_tools_list"])
    assert updated_state['retrieved_documents'] == state['retrieved_documents']
    assert updated_state['file_contents'] == state['file_contents']
    assert updated_state['current_step_index'] == 1 

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for response_synthesizer_node")
def test_response_synthesizer_greeting(node_test_setup):
    state = create_initial_agent_state(user_query="Hi", query_type="greeting")
    updated_state = response_synthesizer_node(node_test_setup["default_config"], state)
    assert "Hello! I am a RAG agent." in updated_state['final_answer']

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for response_synthesizer_node")
def test_response_synthesizer_ambiguous(node_test_setup):
    state = create_initial_agent_state(
        user_query="Tell me more", 
        query_type="ambiguous",
        missing_info_request="More about what topic?"
    )
    updated_state = response_synthesizer_node(node_test_setup["default_config"], state)
    assert state['missing_info_request'] == updated_state['final_answer']

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for response_synthesizer_node")
def test_response_synthesizer_with_data(node_test_setup):
    state = create_initial_agent_state(
        user_query="What is in file.txt and what is LangGraph?", 
        query_type="retrieval_and_file_read", 
        retrieved_documents=[Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.")],
        file_contents={"file.txt": "This file contains important notes."}
    )
    updated_state = response_synthesizer_node(node_test_setup["default_config"], state)
    assert "LangGraph is a library" in updated_state['final_answer']
    assert "important notes" in updated_state['final_answer']

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for response_synthesizer_node")
def test_response_synthesizer_no_data_for_query(node_test_setup):
    state = create_initial_agent_state(
        user_query="What is the color of the sky on Mars if there are no documents?", 
        query_type="retrieval",
        retrieved_documents=[], 
        file_contents={}
    )
    updated_state = response_synthesizer_node(node_test_setup["default_config"], state)
    assert "could not find information" in updated_state['final_answer'].lower() or \
           "don't have specific information" in updated_state['final_answer'].lower()

@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for response_synthesizer_node")
def test_response_synthesizer_error_in_file_read(node_test_setup):
    state = create_initial_agent_state(
        user_query="Read missing.txt",
        query_type="file_read",
        file_contents={"missing.txt": "Error: File not found at path: missing.txt"}
    )
    updated_state = response_synthesizer_node(node_test_setup["default_config"], state)
    assert "Error: File not found at path: missing.txt" in updated_state['final_answer']
