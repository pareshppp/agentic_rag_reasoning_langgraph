import os
import functools
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StatefulGraph, END

# Attempt to import AgentState and other utils, trying different paths
try:
    from my_agent.utils.state import AgentState
    from my_agent.utils.nodes import (
        analyze_query_node,
        planner_node,
        tool_executor_node,
        response_synthesizer_node
    )
    from my_agent.utils.tools import faiss_retriever_tool, file_read_tool
except ImportError:
    # Fallback for local execution if my_agent is not in PYTHONPATH directly
    # but the script is run from my-app or similar relative path setup
    print("Attempting relative imports for AgentState, nodes, and tools...")
    from .utils.state import AgentState
    from .utils.nodes import (
        analyze_query_node,
        planner_node,
        tool_executor_node,
        response_synthesizer_node
    )
    from .utils.tools import faiss_retriever_tool, file_read_tool


# 1. Load Environment Variables
load_dotenv()

# 2. Initialize Model and Tools
# Ensure GOOGLE_API_KEY is available for ChatGoogleGenerativeAI
# Note: The nodes themselves initialize their own LLM instance as per nodes.py.
# This central 'llm' instance is not strictly necessary here unless the graph logic itself
# directly uses an LLM, which it currently does not.
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0) 

tools = [faiss_retriever_tool, file_read_tool]

# 3. Graph Definition
graph_builder = StatefulGraph(AgentState)

# 4. Add Nodes to Graph
graph_builder.add_node("query_analyzer", analyze_query_node)
graph_builder.add_node("planner", planner_node)

# Partial function for tool_executor_node to include tools
tool_executor_with_tools = functools.partial(tool_executor_node, tools=tools)
graph_builder.add_node("tool_executor", tool_executor_with_tools)

graph_builder.add_node("response_synthesizer", response_synthesizer_node)

# 5. Set Entry Point
graph_builder.set_entry_point("query_analyzer")

# 6. Define Conditional Edges

# From Query Analyzer
def should_route_to_planner(state: AgentState) -> str:
    """
    Determines whether to route to the planner or directly to the synthesizer
    based on the query type.
    """
    query_type = state.get('query_type')
    print(f"Router (after Query Analyzer): Query Type is '{query_type}'")
    # Actionable types that need planning.
    # 'complex_qa' was in planner's skip list, so it means it's a type that needs planning.
    actionable_types = ['retrieval', 'file_read', 'info_response', 'complex_qa'] 
    
    if query_type in actionable_types:
        print("  Routing to planner.")
        return "planner"
    else: # 'greeting', 'ambiguous' (if no clarification provided yet), 'error', 'other'
        print("  Routing to response_synthesizer.")
        return "response_synthesizer"

graph_builder.add_conditional_edges(
    "query_analyzer",
    should_route_to_planner,
    {
        "planner": "planner",
        "response_synthesizer": "response_synthesizer"
    }
)

# From Planner
def should_route_to_tool_executor_or_synthesizer(state: AgentState) -> str:
    """
    Determines whether to execute tools based on the plan or go to synthesis.
    """
    plan = state.get('plan', [])
    query_type = state.get('query_type') # For logging/context
    print(f"Router (after Planner): Query Type '{query_type}', Current plan: {plan}")
    if plan and len(plan) > 0:
        # If the plan exists and has steps, execute it.
        # The tool_executor can handle "Synthesize answer..." as a step by advancing.
        print("  Routing to tool_executor.")
        return "tool_executor"
    else:
        # No plan, or plan is empty (e.g., for query types planner skipped, like 'greeting')
        print("  Routing to response_synthesizer.")
        return "response_synthesizer"

graph_builder.add_conditional_edges(
    "planner",
    should_route_to_tool_executor_or_synthesizer,
    {
        "tool_executor": "tool_executor",
        "response_synthesizer": "response_synthesizer"
    }
)

# From Tool Executor
def should_continue_tool_execution_or_synthesize(state: AgentState) -> str:
    """
    Determines whether to continue tool execution or proceed to synthesis.
    """
    plan = state.get('plan', [])
    current_step_index = state.get('current_step_index', 0)
    print(f"Router (after Tool Executor): Plan: {plan}, Current Step Index: {current_step_index}")

    if plan and current_step_index < len(plan):
        current_step = plan[current_step_index]
        # Check if the *next* step in the plan is to synthesize.
        # The tool_executor_node itself handles the "Synthesize answer..." step by advancing
        # current_step_index. So, if current_step_index points to "Synthesize answer...",
        # it means the tool_executor has just processed it, and we should now go to the synthesizer.
        # However, the prompt asks to check current_step.lower().startswith("synthesize answer")
        # This implies the *current* step to be executed is synthesis.
        # The tool_executor_node already has logic:
        # elif current_step.lower().startswith("synthesize answer"):
        #    state['current_step_index'] = current_step_index + 1
        #    state['intermediate_steps'].append((current_step, "No tool execution, proceeding to synthesis."))
        #    return state -> THIS IS IN THE NODE, NOT THE ROUTER.
        # So the router here should decide based on whether the *plan is finished* or if the *next specific step is synthesis*.
        # The tool executor advances the step. If it advanced *past* "Synthesize answer...", then plan is effectively done for tools.

        # If the current step (that was just executed, or is about to be if an error occurred and we re-evaluate)
        # is "Synthesize answer...", it means we are ready for the response_synthesizer node.
        # The tool_executor_node itself doesn't execute "Synthesize answer...", it just prepares for it.
        # Let's refine: if the current_step_index points to a "Synthesize answer..." step, it implies
        # previous tool actions are done.
        
        # Simpler logic: The tool_executor_node increments current_step_index *after* execution.
        # So, if current_step_index is now equal to len(plan), all steps are done.
        # Or, if the *next* step to be executed (plan[current_step_index]) is "Synthesize answer...", then go to synthesizer.
        # The tool_executor_node itself skips "Synthesize answer..." and increments the index.
        # So, if tool_executor_node just finished a tool call, and the *next* step is "Synthesize answer...",
        # it will increment current_step_index. The router then sees current_step_index pointing to "Synthesize answer...".
        
        if plan[current_step_index].lower().startswith("synthesize answer"):
             print("  Routing to response_synthesizer (next step is synthesis).")
             return "response_synthesizer"
        else:
             print("  Routing back to tool_executor for next tool action.")
             return "tool_executor" # Continue with next tool call in plan
    else: # Plan complete (current_step_index >= len(plan)) or no plan
        print("  Routing to response_synthesizer (plan complete or no plan).")
        return "response_synthesizer"

graph_builder.add_conditional_edges(
    "tool_executor",
    should_continue_tool_execution_or_synthesize,
    {
        "tool_executor": "tool_executor",
        "response_synthesizer": "response_synthesizer"
    }
)

# 7. Add Edge from Response Synthesizer to END
graph_builder.add_edge("response_synthesizer", END)

# 8. Compile the Graph
agent_runnable = graph_builder.compile()
print("Agent graph compiled successfully.")


# 9. Optional: if __name__ == '__main__': block for testing
if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nError: GOOGLE_API_KEY not found. Please set it in your .env file to run agent tests.")
    else:
        print("\nAgent Runnable Created. You can test invoking it here.")
        print("Note: Ensure dummy files/documents are set up if testing tool execution paths.")

        # Helper to create initial AgentState with all necessary keys
        def create_initial_agent_state_for_test(user_query: str) -> AgentState:
            return AgentState(
                user_query=user_query, chat_history=[], query_type="", plan=[],
                current_step_index=0, retrieved_documents=[], file_contents={},
                final_answer="", missing_info_request="", intermediate_steps=[]
            )

        # --- Test Case 1: Simple Greeting ---
        print("\n--- Test Case: Greeting ---")
        initial_state_greeting = create_initial_agent_state_for_test(user_query="Hello, how are you?")
        
        final_greeting_state = {}
        try:
            for event_index, event_output in enumerate(agent_runnable.stream(initial_state_greeting, {"recursion_limit": 10})):
                print(f"\n--- Event {event_index + 1} ---")
                # event_output is a dictionary where keys are node names and values are their output AgentState
                for node_name, output_state in event_output.items():
                    if node_name == END: # Check if it's the end state
                        print(f"Node '{END}': Final output received.")
                        final_greeting_state = output_state # The final state from the graph
                        break 
                    print(f"Node '{node_name}' output:")
                    if isinstance(output_state, dict): # Each node's output is an AgentState (TypedDict)
                        print(f"  Query Type: {output_state.get('query_type')}")
                        print(f"  Missing Info: {output_state.get('missing_info_request')}")
                        print(f"  Plan: {output_state.get('plan')}")
                        print(f"  Current Step Index: {output_state.get('current_step_index')}")
                        print(f"  Final Answer: {output_state.get('final_answer')}")
                if END in event_output: # Break outer loop if END was processed
                    break
            else: 
                 print("Warning: Graph execution might not have reached END within recursion limit.")
            
            print("\n--- Final State (Greeting) ---")
            print(f"User Query: {initial_state_greeting['user_query']}")
            print(f"Final Answer: {final_greeting_state.get('final_answer')}")
            print(f"Query Type: {final_greeting_state.get('query_type')}")
            
            # Basic assertion
            if not (final_greeting_state.get('final_answer') and "Hello! I am a RAG agent." in final_greeting_state.get('final_answer')):
                 print(f"AssertionError: Greeting test failed. Expected specific greeting, got: {final_greeting_state.get('final_answer')}")

        except Exception as e:
            print(f"Error during greeting test: {e}")


        # --- Test Case 2: File Read (requires dummy file & GOOGLE_API_KEY for planner) ---
        print("\n\n--- Test Case: File Read (Full Flow) ---")
        dummy_file_for_agent_test = "agent_test_dummy_file.txt"
        DUMMY_FILE_CONTENT_AGENT = "This is content from the agent's FileReadTool test."
        final_file_read_state = {}
        try:
            with open(dummy_file_for_agent_test, "w") as f:
                f.write(DUMMY_FILE_CONTENT_AGENT)
            
            initial_state_file_read = create_initial_agent_state_for_test(
                user_query=f"Please read the file '{dummy_file_for_agent_test}' and tell me its content."
            )
            
            for event_index, event_output in enumerate(agent_runnable.stream(initial_state_file_read, {"recursion_limit": 15})):
                print(f"\n--- Event {event_index + 1} (File Read) ---")
                for node_name, output_state in event_output.items():
                    if node_name == END:
                        print(f"Node '{END}': Final output received.")
                        final_file_read_state = output_state
                        break
                    print(f"Node '{node_name}' output:")
                    if isinstance(output_state, dict):
                        print(f"  Query Type: {output_state.get('query_type')}")
                        print(f"  File Contents: {output_state.get('file_contents')}")
                        print(f"  Plan: {output_state.get('plan')}")
                        print(f"  Current Step Index: {output_state.get('current_step_index')}")
                        print(f"  Retrieved Docs: {len(output_state.get('retrieved_documents', []))}")
                        print(f"  Final Answer (so far): {output_state.get('final_answer')}")
                if END in event_output:
                    break
            else:
                print("Warning: Graph execution (File Read) might not have reached END.")

            print("\n--- Final State (File Read) ---")
            print(f"User Query: {initial_state_file_read['user_query']}")
            print(f"Final Answer: {final_file_read_state.get('final_answer')}")
            
            # Basic assertion
            if not (final_file_read_state.get('final_answer') and DUMMY_FILE_CONTENT_AGENT in final_file_read_state.get('final_answer')):
                print(f"AssertionError: File read test failed. Expected '{DUMMY_FILE_CONTENT_AGENT}' in answer, got: {final_file_read_state.get('final_answer')}")

        except Exception as e:
            print(f"Error during file read test: {e}")
        finally:
            if os.path.exists(dummy_file_for_agent_test):
                os.remove(dummy_file_for_agent_test)
            print(f"Cleaned up {dummy_file_for_agent_test}")
        
        print("\nAgent testing block finished. More comprehensive tests should be in a dedicated test file using unittest framework.")
