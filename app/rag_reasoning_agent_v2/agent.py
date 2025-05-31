import os
import functools
import typing
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS # For dummy vector store in main

# Attempt to import V2 specific modules
try:
    from .utils.state import AgentStateV2
    from .utils.config import AgentConfigurationV2
    from .utils.nodes import reasoning_node, tool_executor_node_v2, route_after_reasoning
    from .utils.tools import create_faiss_retriever_tool_v2, create_file_read_tool_v2, DUMMY_DOCS
except ImportError:
    # Fallback for testing or direct execution scenarios
    print("Attempting relative imports for V2 AgentState, config, nodes, and tool factories...")
    # This assumes that if run directly, the script is in a place where 'utils' is a sibling directory
    # or PYTHONPATH is configured. For package structure, the '.' imports are preferred.
    from utils.state import AgentStateV2
    from utils.config import AgentConfigurationV2
    from utils.nodes import reasoning_node, tool_executor_node_v2, route_after_reasoning
    from utils.tools import create_faiss_retriever_tool_v2, create_file_read_tool_v2, DUMMY_DOCS


# 1. Load Environment Variables
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY not found. Agent will likely fail to initialize LLM or Embeddings.")

# 2. Graph Creation Logic
def create_agent_v2_runnable(
    config: AgentConfigurationV2,
    vector_store: typing.Optional[FAISS] = None # Allow passing a vector store, e.g., for testing or pre-initialized stores
):
    """
    Creates and compiles the RAG Reasoning Agent V2 graph runnable.
    """
    print(f"Creating RAG Reasoning Agent V2 graph with config: LLM={config.llm_model_name}, FAISS Path='{config.faiss_index_path}'")

    # Initialize tools
    # The retriever tool will use config.faiss_index_path if vector_store is None.
    retriever_tool = create_faiss_retriever_tool_v2(config, vector_store=vector_store)
    file_read_tool = create_file_read_tool_v2(config) # file_read_tool doesn't need vector_store directly
    
    tools = [retriever_tool, file_read_tool]
    tools_map = {tool.name: tool for tool in tools} # Used by tool_executor_node_v2
    print(f"  Initialized tools: {[tool.name for tool in tools]}")

    # Initialize LLM and bind tools for the reasoning_node
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default,
    )
    llm_with_tools = llm.bind_tools(tools) # This LLM instance is passed to reasoning_node
    print(f"  LLM '{config.llm_model_name}' bound with tools for reasoning_node.")

    # Instantiate StateGraph for AgentStateV2
    graph_builder = StateGraph(AgentStateV2)

    # Bind arguments to node functions using functools.partial
    # LangGraph calls node functions with `state` as the first argument.
    
    # reasoning_node(state: AgentStateV2, agent_config: AgentConfigurationV2, llm_with_tools: ChatGoogleGenerativeAI)
    bound_reasoning_node = functools.partial(reasoning_node, agent_config=config, llm_with_tools=llm_with_tools)
    
    # tool_executor_node_v2(state: AgentStateV2, tools_map: typing.Dict[str, Tool])
    bound_tool_executor_node = functools.partial(tool_executor_node_v2, tools_map=tools_map)

    # Add Nodes to Graph
    graph_builder.add_node("reasoner", bound_reasoning_node)
    graph_builder.add_node("tool_executor", bound_tool_executor_node) # Name matches routing logic
    print("  Nodes 'reasoner' and 'tool_executor' added to graph.")

    # Set Entry Point
    graph_builder.set_entry_point("reasoner")
    print("  Entry point set to 'reasoner'.")

    # Define Conditional Edge from reasoner
    # route_after_reasoning(state: AgentStateV2) -> str:
    # Returns "tool_executor" or END (imported from langgraph.graph)
    graph_builder.add_conditional_edges(
        "reasoner",
        route_after_reasoning, # This function decides the next step
        {
            "tool_executor": "tool_executor", # Key matches return value from route_after_reasoning
            END: END  # If route_after_reasoning returns END, graph terminates
        }
    )
    print("  Conditional edge from 'reasoner' to 'tool_executor' or END defined.")

    # Add Edge from tool_executor back to reasoner for multi-hop capabilities
    graph_builder.add_edge("tool_executor", "reasoner")
    print("  Edge from 'tool_executor' back to 'reasoner' added for loops.")

    # Compile the Graph
    compiled_graph = graph_builder.compile()
    print("  RAG Reasoning Agent V2 graph compiled successfully.")
    return compiled_graph

# 3. Create a default AgentConfigurationV2 instance
# This will use default values specified in AgentConfigurationV2, including faiss_index_path
default_config_v2 = AgentConfigurationV2()
print(f"\nDefault RAG Agent V2 configuration loaded: LLM={default_config_v2.llm_model_name}, FAISS Path='{default_config_v2.faiss_index_path}'")

# 4. Create the main agent_v2_runnable to be exported, using the default config
# The retriever tool will attempt to load or create its FAISS index based on default_config_v2.faiss_index_path
agent_v2_runnable = create_agent_v2_runnable(default_config_v2)
print("Default agent_v2_runnable created and ready for use.")


# Main execution block for simple, direct testing of this script
if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nError: GOOGLE_API_KEY not found. Please set it to run a test invocation.")
    else:
        print("\n--- Testing RAG Reasoning Agent V2 (Direct Script Run) ---")
        
        # For this direct test, we'll use a temporary FAISS index path to avoid conflicts
        # and ensure the DUMMY_DOCS are used.
        temp_faiss_index_dir = f"./temp_faiss_index_v2_main_test_{os.getpid()}"
        
        # Ensure the directory for the FAISS index is clean/exists
        if os.path.exists(temp_faiss_index_dir):
            import shutil
            try:
                shutil.rmtree(temp_faiss_index_dir)
                print(f"  Cleaned up existing temp FAISS directory: {temp_faiss_index_dir}")
            except OSError as e:
                print(f"  Error cleaning up temp FAISS directory {temp_faiss_index_dir}: {e}")
        os.makedirs(temp_faiss_index_dir, exist_ok=True)
        
        # Specific config for this test run, pointing to the temp FAISS path
        test_run_config = AgentConfigurationV2(faiss_index_path=os.path.join(temp_faiss_index_dir, "test_v2.faiss"))

        # Create a dummy vector store for this test run and pass it to the agent creator
        # This ensures DUMMY_DOCS are used and the tool doesn't try to load from a potentially empty default path.
        try:
            test_embeddings = GoogleGenerativeAIEmbeddings(model=test_run_config.embedding_model_name)
            test_vector_store = FAISS.from_documents(DUMMY_DOCS, test_embeddings)
            # The create_faiss_retriever_tool_v2 will use this in-memory store if provided,
            # otherwise it would try to load from test_run_config.faiss_index_path.
            # For this test, providing it directly is cleaner.
            print(f"  Test FAISS vector store created in-memory with DUMMY_DOCS for this run.")
            
            test_agent_runnable = create_agent_v2_runnable(test_run_config, vector_store=test_vector_store)
            print(f"  Test agent runnable created successfully for this run.")

        except Exception as e:
            print(f"  Error creating test vector store or test agent runnable: {e}")
            import traceback
            traceback.print_exc()
            test_agent_runnable = None 

        if test_agent_runnable:
            # Create a dummy file for FileReadTool to use during the test
            dummy_file_path = "test_dummy_file_v2_main.txt"
            with open(dummy_file_path, "w") as f:
                f.write("This is a test dummy file for RAG Reasoning Agent V2, created during main script execution. It contains simple text about testing.")
            print(f"  Test dummy file created at '{dummy_file_path}'.")

            # Test Queries
            queries_to_test = [
                "What is LangGraph based on the provided documents?", # Should use retriever with DUMMY_DOCS
                f"What content is in the file named '{dummy_file_path}'?", # Should use file reader
                f"Summarize the concept of LangGraph and also tell me what's in the file '{dummy_file_path}'." # Multi-hop
            ]

            for test_query in queries_to_test:
                print(f"\n--- Invoking agent for query: '{test_query}' ---")
                
                system_prompt_content = (
                    "You are an AI assistant. You have a RetrieverTool for knowledge and a FileReadTool for files. "
                    "Reason step-by-step. If you use a tool, briefly state why. Answer when ready."
                )
                # Initial state for each query
                initial_agent_state = AgentStateV2(
                    user_query=test_query,
                    chat_history=[ # chat_history should contain the initial system prompt and user query
                        SystemMessage(content=system_prompt_content),
                        HumanMessage(content=test_query)
                    ],
                    intermediate_steps=[], # This will be populated by the graph during execution
                    retrieved_documents=[],
                    file_contents={},
                    final_answer=None # Should be None initially
                )
                
                try:
                    # Stream events to observe the agent's flow
                    final_result_state = None
                    for event_chunk_index, event_chunk in enumerate(test_agent_runnable.stream(initial_agent_state, {"recursion_limit": 10})):
                        # event_chunk is a dictionary where keys are node names and values are their outputs (AgentStateV2)
                        print(f"Event Chunk {event_chunk_index + 1}: { {k: v.get('final_answer', 'No final answer yet') if isinstance(v, dict) else v for k,v in event_chunk.items()} }") # Print summary
                        if END in event_chunk: # Check if the graph has reached its designated end state
                             final_result_state = event_chunk[END]
                             break 
                    
                    if final_result_state and final_result_state.get('final_answer'):
                        print(f"\nFinal Answer for '{test_query}':\n{final_result_state['final_answer']}")
                    else:
                        print(f"\nNo final answer obtained for '{test_query}'. Last state: {final_result_state}")
                    
                except Exception as e:
                    print(f"  Error during test invocation for query '{test_query}': {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    print("-" * 60)
            
            # Clean up dummy file and the temporary FAISS directory
            if os.path.exists(dummy_file_path):
                os.remove(dummy_file_path)
                print(f"  Test dummy file '{dummy_file_path}' removed.")
            if os.path.exists(temp_faiss_index_dir):
                import shutil
                try:
                    shutil.rmtree(temp_faiss_index_dir)
                    print(f"  Temp FAISS directory '{temp_faiss_index_dir}' removed.")
                except OSError as e:
                    print(f"  Error removing temp FAISS directory {temp_faiss_index_dir}: {e}")
        else:
            print("  Skipping test invocations as test_agent_runnable could not be created.")
