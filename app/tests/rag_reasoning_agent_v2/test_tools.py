import pytest
import os
import shutil # For directory operations in tests
from unittest.mock import patch, MagicMock

# Attempt to import from the V2 agent's utils
# This structure assumes tests are run from the project root or PYTHONPATH is set up
try:
    from app.rag_reasoning_agent_v2.utils.tools import (
        create_file_read_tool_v2,
        create_faiss_retriever_tool_v2,
        DUMMY_DOCS,
        _generate_sub_queries_configured_v2 # For more specific testing if needed
    )
    from app.rag_reasoning_agent_v2.utils.config import AgentConfigurationV2
except ImportError:
    # Fallback for scenarios where the path might be different (e.g. CI)
    # This might require adjusting PYTHONPATH if tests are not found
    print("Attempting fallback imports for test_tools.py - ensure PYTHONPATH is correct")
    from rag_reasoning_agent_v2.utils.tools import (
        create_file_read_tool_v2,
        create_faiss_retriever_tool_v2,
        DUMMY_DOCS,
        _generate_sub_queries_configured_v2
    )
    from rag_reasoning_agent_v2.utils.config import AgentConfigurationV2

from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Fixtures --- 

@pytest.fixture
def default_v2_config():
    """Provides a default AgentConfigurationV2 instance for tests."""
    return AgentConfigurationV2()

@pytest.fixture
def file_read_tool_v2(default_v2_config):
    """Provides an instance of the FileReadTool for V2."""
    return create_file_read_tool_v2(default_v2_config)

@pytest.fixture
def temp_test_dir(tmp_path_factory):
    """Create a temporary directory for test files and ensure cleanup."""
    temp_dir = tmp_path_factory.mktemp("test_files_")
    yield temp_dir
    # shutil.rmtree(temp_dir) # tmp_path_factory handles cleanup

# --- Tests for FileReadTool (V2) --- 

def test_file_read_tool_success(file_read_tool_v2, temp_test_dir):
    """Test successful reading of a file."""
    file_path = temp_test_dir / "test_file.txt"
    expected_content = "Hello, this is a test file!"
    with open(file_path, "w") as f:
        f.write(expected_content)
    
    content = file_read_tool_v2.invoke(str(file_path))
    assert content == expected_content

def test_file_read_tool_non_existent(file_read_tool_v2, temp_test_dir):
    """Test reading a non-existent file."""
    non_existent_path = temp_test_dir / "non_existent.txt"
    content = file_read_tool_v2.invoke(str(non_existent_path))
    assert "Error: File not found" in content

def test_file_read_tool_is_directory(file_read_tool_v2, temp_test_dir):
    """Test attempting to read a directory."""
    # temp_test_dir itself is a directory
    content = file_read_tool_v2.invoke(str(temp_test_dir))
    assert "Error: Expected a file path, but got a directory" in content

def test_file_read_tool_restricted_env_file(file_read_tool_v2, temp_test_dir):
    """Test attempting to read a restricted .env file."""
    restricted_file_path = temp_test_dir / ".env.test"
    with open(restricted_file_path, "w") as f:
        f.write("SECRET_KEY=123")
    
    content = file_read_tool_v2.invoke(str(restricted_file_path))
    assert "Error: Access to this file path" in content
    assert "is restricted" in content

def test_file_read_tool_restricted_pyc_file(file_read_tool_v2, temp_test_dir):
    """Test attempting to read a restricted .pyc file."""
    restricted_file_path = temp_test_dir / "test_module.pyc"
    with open(restricted_file_path, "wb") as f: # .pyc are binary
        f.write(b"\x00\x00\x00\x00") # Dummy binary content
    
    content = file_read_tool_v2.invoke(str(restricted_file_path))
    assert "Error: Access to this file path" in content
    assert "is restricted" in content

# --- Fixtures for RetrieverTool (V2) --- 

@pytest.fixture
def test_embeddings(default_v2_config: AgentConfigurationV2):
    """Provides GoogleGenerativeAIEmbeddings instance for tests."""
    # Ensure GOOGLE_API_KEY is available for this, or mock if needed in CI without keys
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not found, skipping embedding-dependent tests.")
    return GoogleGenerativeAIEmbeddings(model=default_v2_config.embedding_model_name)

@pytest.fixture
def test_vector_store(test_embeddings):
    """Provides an in-memory FAISS vector store with DUMMY_DOCS."""
    try:
        return FAISS.from_documents(DUMMY_DOCS, test_embeddings)
    except Exception as e:
        pytest.skip(f"Failed to create FAISS vector store, possibly due to API key or network: {e}")

@pytest.fixture
def retriever_tool_v2(default_v2_config: AgentConfigurationV2, test_vector_store: FAISS):
    """Provides an instance of the RetrieverTool for V2, using a test vector store."""
    return create_faiss_retriever_tool_v2(config=default_v2_config, vector_store=test_vector_store)

@pytest.fixture
def retriever_tool_v2_no_store(default_v2_config: AgentConfigurationV2):
    """Provides an instance of the RetrieverTool for V2 with no vector store."""
    return create_faiss_retriever_tool_v2(config=default_v2_config, vector_store=None)

# --- Tests for RetrieverTool (V2) --- 

def test_retriever_tool_v2_success(retriever_tool_v2: Tool):
    """Test successful document retrieval using the V2 retriever tool."""
    query = "What is LangGraph?"
    # Ensure GOOGLE_API_KEY is available for the sub-query LLM call, or mock if needed
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not found, skipping retriever tool test that may call LLM.")
        
    retrieved_docs = retriever_tool_v2.invoke(query)
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) > 0
    assert any("LangGraph" in doc.page_content for doc in retrieved_docs)

def test_retriever_tool_v2_no_vector_store(retriever_tool_v2_no_store: Tool):
    """Test retriever tool behavior when the vector store is not provided."""
    query = "What is LangGraph?"
    retrieved_docs = retriever_tool_v2_no_store.invoke(query)
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) == 1 # Expecting a single Document with an error message
    assert "Error: FAISS vector store not provided or initialized." in retrieved_docs[0].page_content

def test_retriever_tool_v2_sub_queries_single(default_v2_config: AgentConfigurationV2, test_vector_store: FAISS):
    """Test sub-query generation when max_sub_queries is 1 (should use original query)."""
    # Modify config for this specific test
    config = default_v2_config.model_copy(update={"max_sub_queries": 1})
    
    # Mock the LLM used for sub-query generation to ensure it's not actually called
    # or to verify it's called with the expected prompt if we wanted to go deeper.
    # For max_sub_queries=1, the _generate_sub_queries_configured_v2 should ideally just return the original query.
    mock_llm_for_sub_queries = MagicMock(spec=ChatGoogleGenerativeAI)
    # If _generate_sub_queries_configured_v2 is called, make it return the original query directly
    # This tests the path where the LLM might not be strictly needed for max_sub_queries=1
    
    query = "Tell me about FAISS library."

    # We need to patch the ChatGoogleGenerativeAI instantiation within create_faiss_retriever_tool_v2
    # or more directly, patch _generate_sub_queries_configured_v2 if it's easier to isolate.

    # Simpler approach: Test _generate_sub_queries_configured_v2 directly for this case.
    # This assumes the LLM for sub-queries is correctly instantiated in the tool creation.
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not found, skipping sub-query test that may call LLM.")

    # The actual LLM for sub-queries in the tool
    llm_for_sub_queries_in_tool = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default
    )

    # Test the internal sub-query generation function
    generated_sub_queries = _generate_sub_queries_configured_v2(
        query=query, 
        config=config, 
        llm_instance=llm_for_sub_queries_in_tool
    )
    
    assert len(generated_sub_queries) == 1
    # Depending on the LLM's rephrasing for a single query, it might be slightly different.
    # For simplicity, we check if the core term is present.
    # A more robust mock would control the LLM output precisely.
    assert "FAISS" in generated_sub_queries[0] 

    # Now test the full tool with this config
    retriever_tool_with_single_subquery_config = create_faiss_retriever_tool_v2(
        config=config, 
        vector_store=test_vector_store
    )
    retrieved_docs = retriever_tool_with_single_subquery_config.invoke(query)
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) > 0
    assert any("FAISS" in doc.page_content for doc in retrieved_docs)
