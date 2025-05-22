import os
import sys
import unittest
import json # Not strictly needed for these tests, but good for general test files
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from my_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # For chat history
from langchain_core.documents import Document
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import (
    analyze_query_node,
    planner_node,
    tool_executor_node,
    response_synthesizer_node
)
# Tools are needed for tool_executor_node tests
from my_agent.utils.tools import faiss_retriever_tool, file_read_tool

load_dotenv()

# Helper to create initial AgentState with all necessary keys
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

# Dummy file for testing FileReadTool via ToolExecutorNode
DUMMY_EXECUTOR_FILE_PATH = "dummy_executor_test_file.txt"
DUMMY_EXECUTOR_CONTENT = "This is content from the executor test dummy file for nodes test."

def setup_dummy_executor_file():
    with open(DUMMY_EXECUTOR_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_EXECUTOR_CONTENT)

def teardown_dummy_executor_file():
    if os.path.exists(DUMMY_EXECUTOR_FILE_PATH):
        os.remove(DUMMY_EXECUTOR_FILE_PATH)

class TestAgentNodes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.google_api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
        if not cls.google_api_key_present:
            print("WARNING: GOOGLE_API_KEY not found. Skipping API-dependent node tests.")

    # --- Tests for analyze_query_node ---
    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for analyze_query_node")
    def test_01_analyze_query_greeting(self):
        state = create_initial_agent_state(user_query="Hello there!")
        updated_state = analyze_query_node(state)
        self.assertEqual(updated_state['query_type'], 'greeting', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for analyze_query_node")
    def test_02_analyze_query_file_read(self):
        state = create_initial_agent_state(user_query="Can you read the file named report.txt and also check /tmp/data.csv?")
        updated_state = analyze_query_node(state)
        self.assertEqual(updated_state['query_type'], 'file_read',  f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")
        self.assertIn("report.txt", updated_state['file_contents'])
        self.assertIn("/tmp/data.csv", updated_state['file_contents'])

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for analyze_query_node")
    def test_03_analyze_query_ambiguous(self):
        history = [AIMessage(content="I can help with information retrieval and file processing.")]
        state = create_initial_agent_state(user_query="Tell me more.", chat_history=history)
        updated_state = analyze_query_node(state)
        self.assertEqual(updated_state['query_type'], 'ambiguous', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")
        self.assertTrue(len(updated_state['missing_info_request']) > 0)

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for analyze_query_node")
    def test_04_analyze_query_retrieval(self):
        state = create_initial_agent_state(user_query="What is the capital of France?")
        updated_state = analyze_query_node(state)
        self.assertEqual(updated_state['query_type'], 'retrieval', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for analyze_query_node")
    def test_05_analyze_query_info_response(self):
        history = [
            HumanMessage(content="Can you check the config?"),
            AIMessage(content="Which configuration file are you referring to?")
        ]
        state = create_initial_agent_state(
            user_query="Yes, I meant the main configuration file for the web server.",
            chat_history=history,
            missing_info_request="Which configuration file are you referring to?" # from previous turn
        )
        updated_state = analyze_query_node(state)
        self.assertEqual(updated_state['query_type'], 'info_response', f"Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")

    # --- Tests for planner_node ---
    def test_06_planner_greeting(self): # Should skip planning
        state = create_initial_agent_state(user_query="Hello", query_type="greeting")
        updated_state = planner_node(state)
        self.assertEqual(len(updated_state['plan']), 0)

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for planner_node with LLM")
    def test_07_planner_file_read(self):
        state = create_initial_agent_state(
            user_query="Read report.txt", 
            query_type="file_read",
            file_contents={"report.txt": ""} # From analyzer
        )
        updated_state = planner_node(state)
        self.assertTrue(len(updated_state['plan']) > 0, f"Plan was empty. Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")
        if updated_state['plan']: # Avoid index error if plan is empty
            self.assertIn("Read file 'report.txt'", updated_state['plan'][0])
            self.assertIn("Synthesize answer", updated_state['plan'][-1])

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for planner_node with LLM")
    def test_08_planner_retrieval(self):
        state = create_initial_agent_state(user_query="What is LangGraph?", query_type="retrieval")
        updated_state = planner_node(state)
        self.assertTrue(len(updated_state['plan']) > 0, f"Plan was empty. Reasoning: {updated_state['intermediate_steps'][-1][1].get('reasoning') if updated_state['intermediate_steps'] else 'N/A'}")
        if updated_state['plan']:
            self.assertIn("Use retriever for", updated_state['plan'][0])
            self.assertIn("Synthesize answer", updated_state['plan'][-1])
            
    # --- Tests for tool_executor_node ---
    def test_09_tool_executor_no_plan(self):
        state = create_initial_agent_state(user_query="N/A", plan=[])
        updated_state = tool_executor_node(state, [file_read_tool])
        self.assertEqual(updated_state['current_step_index'], 0) # No change

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for FAISS tool in tool_executor_node")
    def test_10_tool_executor_faiss_retriever(self):
        state = create_initial_agent_state(
            user_query="What is LangGraph?", 
            query_type="retrieval",
            plan=["Use retriever for 'LangGraph library features'", "Synthesize answer using all gathered information."]
        )
        updated_state = tool_executor_node(state, [faiss_retriever_tool, file_read_tool])
        self.assertTrue(len(updated_state['retrieved_documents']) > 0)
        self.assertIsInstance(updated_state['retrieved_documents'][0], Document)
        self.assertEqual(updated_state['current_step_index'], 1)

    def test_11_tool_executor_file_read_success(self):
        setup_dummy_executor_file()
        state = create_initial_agent_state(
            user_query=f"Read {DUMMY_EXECUTOR_FILE_PATH}",
            query_type="file_read",
            plan=[f"Read file '{DUMMY_EXECUTOR_FILE_PATH}'", "Synthesize answer using all gathered information."],
            file_contents={DUMMY_EXECUTOR_FILE_PATH: ""} # Analyzer adds key
        )
        try:
            updated_state = tool_executor_node(state, [faiss_retriever_tool, file_read_tool])
            self.assertEqual(updated_state['file_contents'][DUMMY_EXECUTOR_FILE_PATH], DUMMY_EXECUTOR_CONTENT)
            self.assertEqual(updated_state['current_step_index'], 1)
        finally:
            teardown_dummy_executor_file()

    def test_12_tool_executor_file_read_not_found(self):
        non_existent_file = "non_existent_for_executor.txt"
        state = create_initial_agent_state(
            user_query=f"Read {non_existent_file}",
            query_type="file_read",
            plan=[f"Read file '{non_existent_file}'", "Synthesize answer using all gathered information."],
            file_contents={non_existent_file: ""}
        )
        updated_state = tool_executor_node(state, [faiss_retriever_tool, file_read_tool])
        self.assertIn("Error: File not found", updated_state['file_contents'][non_existent_file])
        self.assertEqual(updated_state['current_step_index'], 1)

    def test_13_tool_executor_synthesize_step(self):
        state = create_initial_agent_state(
            user_query="N/A",
            plan=["Synthesize answer using all gathered information."],
            current_step_index=0
        )
        updated_state = tool_executor_node(state, []) # No tools needed for synth
        self.assertEqual(updated_state['current_step_index'], 1)
        self.assertIn("No tool execution, proceeding to synthesis", updated_state['intermediate_steps'][-1][1])

    # --- Tests for response_synthesizer_node ---
    def test_14_synthesizer_greeting(self):
        state = create_initial_agent_state(user_query="Hi", query_type="greeting")
        updated_state = response_synthesizer_node(state)
        self.assertIn("Hello! I am a RAG agent.", updated_state['final_answer'])

    def test_15_synthesizer_ambiguous(self):
        state = create_initial_agent_state(
            user_query="Tell me about it.", 
            query_type="ambiguous",
            missing_info_request="Could you please specify what 'it' refers to?"
        )
        updated_state = response_synthesizer_node(state)
        self.assertEqual(updated_state['final_answer'], "Could you please specify what 'it' refers to?")

    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for response_synthesizer_node LLM call")
    def test_16_synthesizer_data_synthesis(self):
        sample_doc = Document(page_content="LangGraph is a library by LangChain.")
        sample_file_content = "FAISS helps with vector search."
        state = create_initial_agent_state(
            user_query="What are LangGraph and FAISS?",
            query_type="retrieval", # or file_read
            retrieved_documents=[sample_doc],
            file_contents={"notes.txt": sample_file_content}
        )
        updated_state = response_synthesizer_node(state)
        self.assertIn("LangGraph", updated_state['final_answer'])
        self.assertIn("FAISS", updated_state['final_answer'])
        # Check for citation (simplified check)
        self.assertTrue("Document 1" in updated_state['final_answer'] or "notes.txt" in updated_state['final_answer'] or "LangChain" in updated_state['final_answer'])
        
    @unittest.skipUnless(google_api_key_present, "GOOGLE_API_KEY required for response_synthesizer_node LLM call")
    def test_17_synthesizer_insufficient_data(self):
        state = create_initial_agent_state(
            user_query="What is the color of the sky on Mars?",
            query_type="retrieval",
            retrieved_documents=[Document(page_content="Earth's sky is blue.")]
        )
        updated_state = response_synthesizer_node(state)
        self.assertTrue(
            "cannot answer" in updated_state['final_answer'].lower() or \
            "based on the provided" in updated_state['final_answer'].lower() or \
            "no relevant information" in updated_state['final_answer'].lower()
        )

if __name__ == '__main__':
    unittest.main()
