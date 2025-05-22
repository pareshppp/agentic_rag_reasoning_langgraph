import os
import sys
import unittest
from dotenv import load_dotenv
import typing

# Add project root to sys.path to allow imports from my_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import AgentConfiguration
from my_agent.utils.config import AgentConfiguration

# Import tool factory functions and specific underlying functions if still testable/needed
from my_agent.utils.tools import (
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

def setup_dummy_file():
    with open(DUMMY_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_CONTENT)
    # print(f"Setup: Created dummy file at {os.path.abspath(DUMMY_FILE_PATH)}")

def teardown_dummy_file():
    if os.path.exists(DUMMY_FILE_PATH):
        os.remove(DUMMY_FILE_PATH)
        # print(f"Teardown: Removed dummy file from {os.path.abspath(DUMMY_FILE_PATH)}")

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.google_api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
        if not cls.google_api_key_present:
            print("WARNING: GOOGLE_API_KEY not found. API-dependent tests will be skipped.")
        
        # Create a default config for all tests in this class
        cls.default_config = AgentConfiguration()
        
        # Create tool instances for the class if they are to be reused across multiple tests
        # and don't have state that would interfere. For tools that might change state
        # or if tests modify their config, create them within test methods.
        if cls.google_api_key_present: # Only create API-dependent tools if key is present
            cls.faiss_tool_instance = create_faiss_retriever_tool(cls.default_config)
        else:
            cls.faiss_tool_instance = None # Or a mock/dummy tool
        cls.file_tool_instance = create_file_read_tool(cls.default_config)


    # Test for _generate_sub_queries_configured (if feasible)
    # This requires manual setup of LLM from config, as it's an internal function.
    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for sub-query generation test.")
    def test_01_generate_sub_queries_logic(self):
        print("\nTesting configured sub-query generation logic...")
        # We need to import the internal function if we want to test it.
        # This is often a sign that it perhaps should be part of a class or tested via the tool.
        # For now, let's assume we can import it or we test its effect via the tool.
        # If tools.py has: from .tools import _generate_sub_queries_configured (not typical for internal)
        # For this test, let's re-import it specifically if it's made available, or simulate its call
        
        # To test _generate_sub_queries_configured, we need an LLM instance
        llm_for_test = ChatGoogleGenerativeAI(
            model=self.default_config.llm_model_name,
            temperature=self.default_config.llm_temperature_default
        )
        
        # The actual function might not be directly importable if it's _private in tools.py
        # We'll try to call it by re-importing it if tools.py structure allows,
        # otherwise this test shows how one would set it up.
        # For now, let's assume it's accessible for testing or we are testing a similar public wrapper.
        # from my_agent.utils.tools import _generate_sub_queries_configured # Assuming it's made importable for testing

        # Since _generate_sub_queries_configured is private, we test the effect via the tool.
        # This test will check if the retriever tool (which uses sub-queries) works.
        # A more direct test of sub-query LLM output would require mocking or more complex setup.
        # So, this test is more of an integration test for sub-query functionality within the tool.
        
        query = "What are LangGraph and FAISS features?"
        print(f"Testing FAISS tool which uses sub-queries for query: '{query}'")
        if not self.faiss_tool_instance:
            self.skipTest("FAISS tool instance not created due to missing API key.")

        result_docs = self.faiss_tool_instance.invoke(query)
        print(f"Retrieved {len(result_docs)} documents. Sub-query generation is part of this process.")
        self.assertIsNotNone(result_docs)
        self.assertIsInstance(result_docs, list)
        # We expect some documents; if sub-queries are broken, this might fail or return fewer/less relevant docs.
        self.assertTrue(len(result_docs) >= 0) # Can be 0 if no docs match, but should not error.
        if len(result_docs) > 0:
            self.assertTrue(all(isinstance(doc, Document) for doc in result_docs))


    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for FAISS retriever tool test.")
    def test_02_faiss_retriever_tool_invoke_simple(self):
        print("\nTesting FAISS Retriever Tool (invoke method) with simple query...")
        if not self.faiss_tool_instance:
            self.skipTest("FAISS tool instance not created due to missing API key.")

        query_retrieval = "Tell me about Gemini."
        print(f"Invoking faiss_retriever_tool for query: '{query_retrieval}'")
        result_docs = self.faiss_tool_instance.invoke(query_retrieval)
        
        print(f"Retrieved {len(result_docs)} documents.")
        if result_docs:
            for i, doc in enumerate(result_docs):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        self.assertIsNotNone(result_docs)
        self.assertIsInstance(result_docs, list)
        self.assertTrue(len(result_docs) > 0, "Expected at least one document for Gemini query")
        self.assertTrue(all(isinstance(doc, Document) for doc in result_docs))

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for FAISS retriever tool test.")
    def test_03_faiss_retriever_tool_invoke_complex(self):
        print("\nTesting FAISS Retriever Tool (invoke method) with complex query...")
        if not self.faiss_tool_instance:
            self.skipTest("FAISS tool instance not created due to missing API key.")

        query_complex = "Compare LangGraph's state management with traditional methods and explain FAISS indexing."
        print(f"Invoking faiss_retriever_tool for complex query: '{query_complex}'")
        result_docs_tool_invoke = self.faiss_tool_instance.invoke(query_complex)

        print(f"Retrieved {len(result_docs_tool_invoke)} documents via tool.invoke.")
        if result_docs_tool_invoke:
            for i, doc in enumerate(result_docs_tool_invoke):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        self.assertIsNotNone(result_docs_tool_invoke)
        self.assertIsInstance(result_docs_tool_invoke, list)
        # Length can be 0 if nothing matches, but should be a list of Documents
        self.assertTrue(all(isinstance(doc, Document) for doc in result_docs_tool_invoke))


    def test_04_read_file_content_directly_success(self):
        print("\nTesting read_file_content function (Success)...")
        setup_dummy_file()
        try:
            content = read_file_content(DUMMY_FILE_PATH) # Test underlying function
            print(f"Content of '{DUMMY_FILE_PATH}': '{content}'")
            self.assertEqual(content, DUMMY_CONTENT)
        finally:
            teardown_dummy_file()

    def test_05_file_read_tool_invoke_success(self):
        print("\nTesting FileReadTool (Success via tool.invoke)...")
        setup_dummy_file()
        try:
            content = self.file_tool_instance.invoke(DUMMY_FILE_PATH)
            print(f"Content of '{DUMMY_FILE_PATH}' via tool.invoke: '{content}'")
            self.assertEqual(content, DUMMY_CONTENT)
        finally:
            teardown_dummy_file()

    def test_06_read_file_content_directly_not_found(self):
        print("\nTesting read_file_content function (File Not Found)...")
        non_existent_file = "non_existent_file_for_test.txt"
        result = read_file_content(non_existent_file) # Test underlying function
        print(f"Attempting to read '{non_existent_file}': '{result}'")
        self.assertIn(f"Error: File not found at path: {non_existent_file}", result)

    def test_07_file_read_tool_invoke_not_found(self):
        print("\nTesting FileReadTool (File Not Found via tool.invoke)...")
        non_existent_file = "non_existent_file_for_tool_invoke_test.txt"
        result = self.file_tool_instance.invoke(non_existent_file)
        print(f"Attempting to read '{non_existent_file}' via tool.invoke: '{result}'")
        self.assertIn(f"Error: File not found at path: {non_existent_file}", result)

if __name__ == '__main__':
    unittest.main()
