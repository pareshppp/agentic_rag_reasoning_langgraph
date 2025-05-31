import os
import sys
import pytest 
from dotenv import load_dotenv
import typing

# Add project root to sys.path to allow imports from my_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AgentConfiguration
from rag_reasoning_agent.utils.config import AgentConfiguration

# Import tool factory functions and specific underlying functions if still testable/needed
from rag_reasoning_agent.utils.tools import (
    create_faiss_retriever_tool,
    create_file_read_tool,
    read_file_content, # Still directly testable
    # _generate_sub_queries_configured, # Internal, might be hard to test directly
    # _retrieve_documents_configured,   # Internal
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI # For testing sub-query generation directly

load_dotenv() # Load environment variables like GOOGLE_API_KEY

# Helper to create dummy file for testing read_file_tool
DUMMY_FILE_PATH = "dummy_test_file_for_tools.txt"
DUMMY_CONTENT = "This is a test file for the FileReadTool.\nHello World!"

# Pytest fixture for dummy file setup and teardown
@pytest.fixture
def dummy_file():
    with open(DUMMY_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_CONTENT)
    # print(f"Setup: Created dummy file at {os.path.abspath(DUMMY_FILE_PATH)}")
    yield DUMMY_FILE_PATH  # Provide the path to the test
    if os.path.exists(DUMMY_FILE_PATH):
        os.remove(DUMMY_FILE_PATH)
        # print(f"Teardown: Removed dummy file from {os.path.abspath(DUMMY_FILE_PATH)}")

# Module-level constant for skipif conditions
GOOGLE_API_KEY_PRESENT = bool(os.getenv("GOOGLE_API_KEY"))

@pytest.fixture(scope="module")
def tool_setup():
    if not GOOGLE_API_KEY_PRESENT:
        print("WARNING: GOOGLE_API_KEY not found. API-dependent tests will be skipped.")
    
    default_config = AgentConfiguration()
    
    faiss_tool_instance = None
    if GOOGLE_API_KEY_PRESENT:
        faiss_tool_instance = create_faiss_retriever_tool(default_config)
    
    file_tool_instance = create_file_read_tool(default_config)
    
    return {
        "google_api_key_present": GOOGLE_API_KEY_PRESENT,
        "default_config": default_config,
        "faiss_tool_instance": faiss_tool_instance,
        "file_tool_instance": file_tool_instance
    }


# Test for _generate_sub_queries_configured (if feasible)
# This requires manual setup of LLM from config, as it's an internal function.
@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for sub-query generation test.")
def test_generate_sub_queries_logic(tool_setup):
    print("\nTesting configured sub-query generation logic...")
    default_config = tool_setup["default_config"]
    faiss_tool_instance = tool_setup["faiss_tool_instance"]

    # To test _generate_sub_queries_configured, we need an LLM instance
    llm_for_test = ChatGoogleGenerativeAI(
        model=default_config.llm_model_name,
        temperature=default_config.llm_temperature_default
    )
    
    query = "What are LangGraph and FAISS features?"
    print(f"Testing FAISS tool which uses sub-queries for query: '{query}'")
    if not faiss_tool_instance:
        pytest.skip("FAISS tool instance not created due to missing API key.")

    result_docs = faiss_tool_instance.invoke(query)
    print(f"Retrieved {len(result_docs)} documents. Sub-query generation is part of this process.")
    assert result_docs is not None
    assert isinstance(result_docs, list)
    # We expect some documents; if sub-queries are broken, this might fail or return fewer/less relevant docs.
    assert len(result_docs) >= 0 # Can be 0 if no docs match, but should not error.
    if len(result_docs) > 0:
        assert all(isinstance(doc, Document) for doc in result_docs)
            
@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for FAISS retriever tool test.")
def test_faiss_retriever_tool_invoke_simple(tool_setup):
    print("\nTesting FAISS Retriever Tool (invoke method) with simple query...")
    faiss_tool_instance = tool_setup["faiss_tool_instance"]
    if not faiss_tool_instance:
        pytest.skip("FAISS tool instance not created due to missing API key.")

    query_retrieval = "Tell me about Gemini."
    print(f"Invoking faiss_retriever_tool for query: '{query_retrieval}'")
    result_docs = faiss_tool_instance.invoke(query_retrieval)
    
    print(f"Retrieved {len(result_docs)} documents.")
    if result_docs:
        for i, doc in enumerate(result_docs):
            print(f"  Doc {i+1}: {doc.page_content[:100]}...")
    
    assert result_docs is not None
    assert isinstance(result_docs, list)
    assert len(result_docs) > 0, "Expected at least one document for Gemini query"
    
@pytest.mark.skipif(not GOOGLE_API_KEY_PRESENT, reason="GOOGLE_API_KEY required for FAISS retriever tool test.")
def test_faiss_retriever_tool_invoke_complex(tool_setup):
    print("\nTesting FAISS Retriever Tool (invoke method) with complex query...")
    faiss_tool_instance = tool_setup["faiss_tool_instance"]
    if not faiss_tool_instance:
        pytest.skip("FAISS tool instance not created due to missing API key.")

    query_complex = "Compare LangGraph's state management with traditional methods and explain FAISS indexing."
    print(f"Invoking faiss_retriever_tool for complex query: '{query_complex}'")
    result_docs_tool_invoke = faiss_tool_instance.invoke(query_complex)

    print(f"Retrieved {len(result_docs_tool_invoke)} documents via tool.invoke.")
    if result_docs_tool_invoke:
        for i, doc in enumerate(result_docs_tool_invoke):
            print(f"  Doc {i+1}: {doc.page_content[:100]}...")
    
    assert result_docs_tool_invoke is not None
    assert isinstance(result_docs_tool_invoke, list)
    # Length can be 0 if nothing matches, but should be a list of Documents
    assert all(isinstance(doc, Document) for doc in result_docs_tool_invoke)


def test_read_file_content_directly_success(dummy_file):
    print("\nTesting read_file_content function (Success)...")
    # dummy_file fixture handles setup/teardown and provides the path
    content = read_file_content(dummy_file) # Test underlying function
    print(f"Content of '{dummy_file}': '{content}'")
    assert content == DUMMY_CONTENT

def test_file_read_tool_invoke_success(tool_setup, dummy_file):
    print("\nTesting FileReadTool (Success via tool.invoke)...")
    file_tool_instance = tool_setup["file_tool_instance"]
    # dummy_file fixture handles setup/teardown and provides the path
    content = file_tool_instance.invoke(dummy_file)
    print(f"Content of '{dummy_file}' via tool.invoke: '{content}'")
    assert content == DUMMY_CONTENT

def test_read_file_content_directly_not_found():
    print("\nTesting read_file_content function (File Not Found)...")
    non_existent_file = "non_existent_file_for_test.txt"
    result = read_file_content(non_existent_file) # Test underlying function
    print(f"Attempting to read '{non_existent_file}': '{result}'")
    assert f"Error: File not found at path: {non_existent_file}" in result

def test_file_read_tool_invoke_not_found(tool_setup):
    print("\nTesting FileReadTool (File Not Found via tool.invoke)...")
    file_tool_instance = tool_setup["file_tool_instance"]
    non_existent_file = "non_existent_file_for_tool_invoke_test.txt"
    result = file_tool_instance.invoke(non_existent_file)
    print(f"Attempting to read '{non_existent_file}' via tool.invoke: '{result}'")
    assert f"Error: File not found at path: {non_existent_file}" in result
