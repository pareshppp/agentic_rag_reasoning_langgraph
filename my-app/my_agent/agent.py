import os
import functools
from dotenv import load_dotenv

from langgraph.graph import StatefulGraph, END

# Attempt to import necessary modules
try:
    from my_agent.utils.state import AgentState
    from my_agent.utils.config import AgentConfiguration
    from my_agent.utils.nodes import (
        analyze_query_node,
        planner_node,
        tool_executor_node,
        response_synthesizer_node
    )
    from my_agent.utils.tools import (
        create_faiss_retriever_tool,
        create_file_read_tool
    )
except ImportError:
    print("Attempting relative imports for AgentState, config, nodes, and tool factories...")
    from .utils.state import AgentState
    from .utils.config import AgentConfiguration
    from .utils.nodes import (
        analyze_query_node,
        planner_node,
        tool_executor_node,
        response_synthesizer_node
    )
    from .utils.tools import (
        create_faiss_retriever_tool,
        create_file_read_tool
    )

# 1. Load Environment Variables
load_dotenv()

# Conditional routing functions (remain unchanged as they operate on AgentState)
def should_route_to_planner(state: AgentState) -> str:
    query_type = state.get('query_type')
    print(f"Router (after Query Analyzer): Query Type is '{query_type}'")
    actionable_types = ['retrieval', 'file_read', 'info_response', 'complex_qa']
    if query_type in actionable_types:
        print("  Routing to planner.")
        return "planner"
    else:
        print("  Routing to response_synthesizer.")
        return "response_synthesizer"

def should_route_to_tool_executor_or_synthesizer(state: AgentState) -> str:
    plan = state.get('plan', [])
    query_type = state.get('query_type')
    print(f"Router (after Planner): Query Type '{query_type}', Current plan: {plan}")
    if plan and len(plan) > 0:
        print("  Routing to tool_executor.")
        return "tool_executor"
    else:
        print("  Routing to response_synthesizer.")
        return "response_synthesizer"

def should_continue_tool_execution_or_synthesize(state: AgentState) -> str:
    plan = state.get('plan', [])
    current_step_index = state.get('current_step_index', 0)
    print(f"Router (after Tool Executor): Plan: {plan}, Current Step Index: {current_step_index}")
    if plan and current_step_index < len(plan):
        if plan[current_step_index].lower().startswith("synthesize answer"):
            print("  Routing to response_synthesizer (next step is synthesis).")
            return "response_synthesizer"
        else:
            print("  Routing back to tool_executor for next tool action.")
            return "tool_executor"
    else:
        print("  Routing to response_synthesizer (plan complete or no plan).")
        return "response_synthesizer"

# 2. Graph Creation Logic
def create_graph_runnable(config: AgentConfiguration):
    """
    Creates and compiles the agent graph runnable with the given configuration.
    """
    print(f"Creating agent graph with configuration: LLM={config.llm_model_name}, Embeddings={config.embedding_model_name}")

    # Initialize tools using factory functions and the provided config
    tools = [
        create_faiss_retriever_tool(config),
        create_file_read_tool(config)  # file_read_tool might not use config but accepts it
    ]
    print(f"  Initialized tools: {[tool.name for tool in tools]}")
    if tools[0].description: # Assuming FAISS tool is first and has a description
        print(f"  FAISS Tool Description (sample): {tools[0].description[:100]}...")


    # Instantiate StatefulGraph
    graph_builder = StatefulGraph(AgentState)

    # Bind config (and tools for executor) to node functions using functools.partial
    analyzer_with_config = functools.partial(analyze_query_node, config)
    planner_with_config = functools.partial(planner_node, config)
    # For tool_executor_node, the signature is (config, state, tools)
    # So, we bind config first, then tools will be the last arg when LangGraph calls it with state.
    # Correction: The state is the first argument provided by LangGraph.
    # The partial should be config, then tools.
    # Original tool_executor_node(config: AgentConfiguration, state: AgentState, tools: typing.List[Tool])
    # LangGraph provides 'state' as the first arg to the node.
    # So, the partial needs to fill in 'config' and 'tools'
    # This means the node function should be callable as: node_func(state, config_val, tools_val)
    # or LangGraph must support passing multiple args.
    # Let's adjust the partial assuming LangGraph calls with `node(state)`.
    # The node itself needs to be `def tool_executor_node(state: AgentState, config: AgentConfiguration, tools: typing.List[Tool])`
    # Or, if LangGraph allows passing other args via `add_node`, that's different.
    # Given the current node signatures (config, state, ...), we need to wrap them
    # so that the first argument is 'state' for LangGraph.

    # Wrapper functions to ensure 'state' is the first argument for LangGraph
    def wrapped_analyze_query_node(state: AgentState):
        return analyze_query_node(config, state)
    
    def wrapped_planner_node(state: AgentState):
        return planner_node(config, state)

    def wrapped_tool_executor_node(state: AgentState):
        return tool_executor_node(config, state, tools=tools)

    def wrapped_response_synthesizer_node(state: AgentState):
        return response_synthesizer_node(config, state)

    # Add Nodes to Graph using wrapped functions
    graph_builder.add_node("query_analyzer", wrapped_analyze_query_node)
    graph_builder.add_node("planner", wrapped_planner_node)
    graph_builder.add_node("tool_executor", wrapped_tool_executor_node)
    graph_builder.add_node("response_synthesizer", wrapped_response_synthesizer_node)
    
    print("  Nodes added to graph.")

    # Set Entry Point
    graph_builder.set_entry_point("query_analyzer")
    print("  Entry point set to 'query_analyzer'.")

    # Define Conditional Edges (using the existing routing functions)
    graph_builder.add_conditional_edges("query_analyzer", should_route_to_planner, {
        "planner": "planner", "response_synthesizer": "response_synthesizer"
    })
    graph_builder.add_conditional_edges("planner", should_route_to_tool_executor_or_synthesizer, {
        "tool_executor": "tool_executor", "response_synthesizer": "response_synthesizer"
    })
    graph_builder.add_conditional_edges("tool_executor", should_continue_tool_execution_or_synthesize, {
        "tool_executor": "tool_executor", "response_synthesizer": "response_synthesizer"
    })
    print("  Conditional edges defined.")

    # Add Edge from Response Synthesizer to END
    graph_builder.add_edge("response_synthesizer", END)
    print("  Edge from response_synthesizer to END added.")

    # Compile the Graph
    compiled_graph = graph_builder.compile()
    print("  Agent graph compiled successfully.")
    return compiled_graph

# 3. Create a default AgentConfiguration instance
default_config = AgentConfiguration()
print(f"\nDefault agent configuration loaded: LLM={default_config.llm_model_name}, Embeddings={default_config.embedding_model_name}")

# 4. Create the main agent_runnable to be exported, using the default config
agent_runnable = create_graph_runnable(default_config)
print("Default agent_runnable created and ready.")

# No if __name__ == '__main__' block, as testing is in test_agent.py
# If basic invocation test is needed here for quick checks:
# if __name__ == '__main__':
#     if not os.getenv("GOOGLE_API_KEY"):
#         print("\nError: GOOGLE_API_KEY not found. Please set it to run a test invocation.")
#     else:
#         print("\n--- Testing agent_runnable with a sample query ---")
#         test_query = "Hello there"
#         initial_state = AgentState(
#             user_query=test_query, chat_history=[], query_type="", plan=[],
#             current_step_index=0, retrieved_documents=[], file_contents={},
#             final_answer="", missing_info_request="", intermediate_steps=[]
#         )
#         try:
#             for event in agent_runnable.stream(initial_state, {"recursion_limit": 10}):
#                 print(f"Event: {event}")
#             # final_state = agent_runnable.invoke(initial_state) # For non-streaming
#             # print(f"Final state for '{test_query}': {final_state.get('final_answer')}")
#         except Exception as e:
#             print(f"Error during test invocation: {e}")
