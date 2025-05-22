import os
import sys
import unittest # Using unittest for structure, though asserts will be simple
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from my_agent
# This assumes 'my-app' is the project root and tests are run from within 'my-app' or 'my-app/tests'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_agent.utils.tools import (
    faiss_retriever_tool, 
    file_read_tool, 
    generate_sub_queries, # This was made accessible in the original tools.py
    retrieve_documents, # Main function for FAISS tool
    read_file_content   # Main function for FileReadTool
)
from langchain_core.documents import Document

load_dotenv() # Load environment variables like GOOGLE_API_KEY

# Helper to create dummy file for testing read_file_tool
DUMMY_FILE_PATH = "dummy_test_file_for_tools.txt"
DUMMY_CONTENT = "This is a test file for the FileReadTool.\nHello World!"

def setup_dummy_file():
    with open(DUMMY_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_CONTENT)
    print(f"Setup: Created dummy file at {os.path.abspath(DUMMY_FILE_PATH)}")

def teardown_dummy_file():
    if os.path.exists(DUMMY_FILE_PATH):
        os.remove(DUMMY_FILE_PATH)
        print(f"Teardown: Removed dummy file from {os.path.abspath(DUMMY_FILE_PATH)}")

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This check runs once before any tests in the class
        if not os.getenv("GOOGLE_API_KEY"):
            print("WARNING: GOOGLE_API_KEY not found in environment variables. Some tests requiring API calls might be skipped or fail.")
            # raise unittest.SkipTest("GOOGLE_API_KEY not set, skipping tests that require it.")

    def test_01_generate_sub_queries_directly(self):
        print("\nTesting generate_sub_queries directly...")
        if not os.getenv("GOOGLE_API_KEY"):
            self.skipTest("GOOGLE_API_KEY not found, skipping direct sub-query generation test.")
        
        query = "What are the features of LangGraph and FAISS?"
        print(f"Generating sub-queries for: '{query}'")
        sub_queries = generate_sub_queries(query)
        print(f"Generated sub-queries: {sub_queries}")
        self.assertIsNotNone(sub_queries)
        self.assertIsInstance(sub_queries, list)
        self.assertTrue(len(sub_queries) > 0) # Expecting at least the original query as fallback
        if len(sub_queries) > 1: # If LLM produced sub-queries
            self.assertTrue(all(isinstance(q, str) for q in sub_queries))

    def test_02_faiss_retriever_tool_retrieval(self):
        print("\nTesting FAISS Retriever Tool (retrieve_documents function)...")
        if not os.getenv("GOOGLE_API_KEY"):
            self.skipTest("GOOGLE_API_KEY not found, skipping FAISS retriever tool test.")

        query_retrieval = "Tell me about LangGraph and Gemini."
        print(f"Retrieving documents for query: '{query_retrieval}'")
        # Test the underlying retrieve_documents function used by the tool
        result_docs = retrieve_documents(query_retrieval) 
        
        print(f"Retrieved {len(result_docs)} documents.")
        if result_docs:
            for i, doc in enumerate(result_docs):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        self.assertIsNotNone(result_docs)
        self.assertIsInstance(result_docs, list)
        # Depending on the sub-query generation and retriever's k value, we might get 1 or more.
        # The dummy docs have info on LangGraph and Gemini.
        self.assertTrue(len(result_docs) > 0, "Expected at least one document for LangGraph/Gemini query") 
        self.assertTrue(all(isinstance(doc, Document) for doc in result_docs))

    def test_03_faiss_retriever_tool_invoke(self):
        print("\nTesting FAISS Retriever Tool (tool.invoke)...")
        if not os.getenv("GOOGLE_API_KEY"):
            self.skipTest("GOOGLE_API_KEY not found, skipping FAISS retriever tool.invoke test.")

        query_complex = "Compare LangGraph's state management with traditional methods and explain FAISS indexing."
        print(f"Invoking faiss_retriever_tool for complex query: '{query_complex}'")
        # Test the tool invocation itself
        result_docs_tool_invoke = faiss_retriever_tool.invoke(query_complex)

        print(f"Retrieved {len(result_docs_tool_invoke)} documents via tool.invoke.")
        if result_docs_tool_invoke:
            for i, doc in enumerate(result_docs_tool_invoke):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        self.assertIsNotNone(result_docs_tool_invoke)
        self.assertIsInstance(result_docs_tool_invoke, list)
        self.assertTrue(len(result_docs_tool_invoke) > 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in result_docs_tool_invoke))

    def test_04_read_file_content_success(self):
        print("\nTesting FileReadTool (Success via read_file_content)...")
        setup_dummy_file()
        try:
            # Test the underlying read_file_content function
            content = read_file_content(DUMMY_FILE_PATH)
            print(f"Content of '{DUMMY_FILE_PATH}': '{content}'")
            self.assertEqual(content, DUMMY_CONTENT)
        finally:
            teardown_dummy_file()

    def test_05_file_read_tool_invoke_success(self):
        print("\nTesting FileReadTool (Success via tool.invoke)...")
        setup_dummy_file()
        try:
            # Test the tool invocation
            content = file_read_tool.invoke(DUMMY_FILE_PATH)
            print(f"Content of '{DUMMY_FILE_PATH}' via tool.invoke: '{content}'")
            self.assertEqual(content, DUMMY_CONTENT)
        finally:
            teardown_dummy_file()

    def test_06_read_file_content_not_found(self):
        print("\nTesting FileReadTool (File Not Found via read_file_content)...")
        non_existent_file = "non_existent_file_for_test.txt"
        # Test the underlying read_file_content function
        result = read_file_content(non_existent_file)
        print(f"Attempting to read '{non_existent_file}': '{result}'")
        self.assertIn(f"Error: File not found at path: {non_existent_file}", result)

    def test_07_file_read_tool_invoke_not_found(self):
        print("\nTesting FileReadTool (File Not Found via tool.invoke)...")
        non_existent_file = "non_existent_file_for_tool_invoke_test.txt"
        # Test the tool invocation
        result = file_read_tool.invoke(non_existent_file)
        print(f"Attempting to read '{non_existent_file}' via tool.invoke: '{result}'")
        self.assertIn(f"Error: File not found at path: {non_existent_file}", result)

if __name__ == '__main__':
    # This allows running the tests directly from this file
    # To run with test discovery (e.g. python -m unittest discover -s tests)
    # this block is not strictly necessary but convenient for direct execution.
    unittest.main()
