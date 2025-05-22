import os
import sys
import unittest
import json # For potential future use, not strictly needed by current agent tests
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from my_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_agent.utils.state import AgentState
from my_agent.utils.config import AgentConfiguration # Import AgentConfiguration
from my_agent.agent import agent_runnable # Import the compiled agent graph, already using default config
from langgraph.graph import END # To check for end state

load_dotenv()

# Dummy file for testing FileReadTool flow via agent
DUMMY_AGENT_TEST_FILE_PATH = "dummy_agent_test_file.txt"
DUMMY_AGENT_TEST_FILE_CONTENT = "This is content from the agent's FileReadTool test, executed via the agent graph."

def setup_dummy_agent_test_file():
    with open(DUMMY_AGENT_TEST_FILE_PATH, "w", encoding='utf-8') as f:
        f.write(DUMMY_AGENT_TEST_FILE_CONTENT)
    # print(f"Setup: Created dummy file for agent test at {os.path.abspath(DUMMY_AGENT_TEST_FILE_PATH)}")

def teardown_dummy_agent_test_file():
    if os.path.exists(DUMMY_AGENT_TEST_FILE_PATH):
        os.remove(DUMMY_AGENT_TEST_FILE_PATH)
        # print(f"Teardown: Removed dummy file for agent test from {os.path.abspath(DUMMY_AGENT_TEST_FILE_PATH)}")

# Helper to create initial AgentState, similar to the one in agent.py's test block
def create_initial_agent_state_for_test(user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query, chat_history=[], query_type="", plan=[],
        current_step_index=0, retrieved_documents=[], file_contents={},
        final_answer="", missing_info_request="", intermediate_steps=[]
    )

class TestLangGraphAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.google_api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
        if not cls.google_api_key_present:
            print("WARNING: GOOGLE_API_KEY not found. API-dependent agent flow tests will be skipped.")
        
        cls.default_config = AgentConfiguration() # Instantiate default config

    @unittest.skipUnless(TestLangGraphAgent.google_api_key_present, "GOOGLE_API_KEY required for agent flows involving LLM calls")
    def test_01_greeting_flow(self):
        print("\n--- Agent Test Case: Greeting Flow ---")
        initial_state = create_initial_agent_state_for_test(user_query="Hello, how are you?")
        final_state = {}
        
        try:
            # agent_runnable is already configured with default_config from agent.py
            for event_index, event_output in enumerate(agent_runnable.stream(initial_state, {"recursion_limit": 10})):
                # print(f"Event {event_index + 1}: {list(event_output.keys())}") 
                if END in event_output:
                    final_state = event_output[END]
                    break
            else:
                self.fail("Graph execution did not reach END within recursion limit for greeting flow.")

            self.assertIsNotNone(final_state.get('final_answer'), "Final answer should exist in the final state.")
            self.assertIn("Hello! I am a RAG agent.", final_state.get('final_answer', ""), "Greeting response not found or incorrect.")
            self.assertEqual(final_state.get('query_type'), "greeting", "Query type should be 'greeting'.")
            # print(f"Final Answer (Greeting): {final_state.get('final_answer')}")
        except Exception as e:
            self.fail(f"Greeting flow test failed with an exception: {e}")

    @unittest.skipUnless(TestLangGraphAgent.google_api_key_present, "GOOGLE_API_KEY required for agent flows involving LLM calls and tools")
    def test_02_file_read_flow(self):
        print("\n--- Agent Test Case: File Read Flow ---")
        setup_dummy_agent_test_file()
        initial_state = create_initial_agent_state_for_test(
            user_query=f"Please read the file '{DUMMY_AGENT_TEST_FILE_PATH}' and tell me its content."
        )
        final_state = {}

        try:
            # agent_runnable is already configured with default_config from agent.py
            for event_index, event_output in enumerate(agent_runnable.stream(initial_state, {"recursion_limit": 15})):
                # print(f"Event {event_index + 1}: {list(event_output.keys())}") 
                if END in event_output:
                    final_state = event_output[END]
                    break
            else:
                self.fail("Graph execution did not reach END within recursion limit for file read flow.")

            self.assertIsNotNone(final_state.get('final_answer'), "Final answer should exist in the final state for file read flow.")
            self.assertIn(DUMMY_AGENT_TEST_FILE_CONTENT, final_state.get('final_answer', ""), 
                          f"Expected dummy file content not found in final answer. Answer was: '{final_state.get('final_answer', '')}'")
            # print(f"Final Answer (File Read): {final_state.get('final_answer')}")

        except Exception as e:
            self.fail(f"File read flow test failed with an exception: {e}")
        finally:
            teardown_dummy_agent_test_file()

    # Example of a test that might use the config for assertions (hypothetical)
    def test_03_check_default_config_values_in_runnable(self):
        # This test is more illustrative, as checking internal config of agent_runnable
        # might be complex. But if agent_runnable had a method like get_config(),
        # or if some behavior clearly reflected a default config value, we could test it.
        # For now, we just ensure default_config is available in the test class.
        print("\n--- Agent Test Case: Check Config Availability ---")
        self.assertIsNotNone(self.default_config, "Default config should be instantiated.")
        self.assertEqual(self.default_config.llm_model_name, "gemini-1.5-pro-latest")
        # print(f"Default LLM model from config: {self.default_config.llm_model_name}")


if __name__ == '__main__':
    unittest.main()
