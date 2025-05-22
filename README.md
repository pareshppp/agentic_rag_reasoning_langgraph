# LangGraph RAG Agent with Gemini

## Overview

This project implements a reasoning-based Retrieval Augmented Generation (RAG) agent using LangGraph, Google's Gemini large language models, and FAISS for vector storage and retrieval. The agent is designed to:

- Understand user queries and classify their intent.
- Create dynamic execution plans to answer queries.
- Utilize tools for information retrieval (from a FAISS vector store) and file reading.
- Synthesize answers based on the gathered information, citing sources where appropriate.
- Be configurable through an `AgentConfiguration` class.

Core Technologies:
- **LangGraph:** For building stateful, multi-actor applications with LLMs.
- **Google Gemini:** Used as the core reasoning and generation model (specifically `gemini-1.5-pro-preview-0506` by default).
- **FAISS:** For efficient similarity search in the vector retriever.
- **Langchain:** Leveraged for components like vector stores, embeddings, and tool abstractions.

## Directory Structure

-   `my-app/`
    -   `my_agent/`: Contains the core agent logic.
        -   `utils/`: Utilities for the agent graph.
            -   `config.py`: Defines `AgentConfiguration` for agent settings.
            -   `tools.py`: Implements tools (FAISS retriever, file reader) and their factory functions.
            -   `nodes.py`: Defines the functions for each node in the LangGraph (query analysis, planning, tool execution, response synthesis).
            -   `state.py`: Defines the `AgentState` TypedDict for the graph.
        -   `agent.py`: Constructs and compiles the LangGraph agent (`agent_runnable`).
    -   `tests/`: Contains unit and integration tests.
        -   `test_tools.py`: Tests for the agent's tools.
        -   `test_nodes.py`: Tests for individual agent nodes.
        -   `test_agent.py`: Tests for the overall agent graph execution.
    -   `.env`: For environment variables (e.g., `GOOGLE_API_KEY`). (Needs to be created by the user).
    -   `requirements.txt`: Python package dependencies.
    -   `langgraph.json`: Configuration for LangGraph Studio integration.
    -   `README.md`: This file.

## Setup Instructions

1.  **Python Version:**
    *   Recommended: Python 3.9 or higher.

2.  **Clone Repository (if applicable):**
    *   If you've obtained this as a standalone project, clone it:
        ```bash
        # git clone <repository_url>
        # cd my-app
        ```

3.  **Create `.env` File:**
    *   In the `my-app` directory, create a file named `.env`.
    *   Add your Google API key to it:
        ```env
        GOOGLE_API_KEY="your_actual_google_api_key_goes_here"
        ```
    *   Replace `"your_actual_google_api_key_goes_here"` with your actual API key.

4.  **Install Dependencies:**
    *   It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

## Running the Agent

### LangGraph Studio (Recommended)

The agent is designed to be easily integrated with LangGraph Studio.
1.  Ensure LangGraph Studio is installed and running.
2.  The `my-app/langgraph.json` file configures the agent graph named "MyRagAgent" to point to `my_agent.agent:agent_runnable`.
3.  You should be able to import or discover this graph within LangGraph Studio and interact with it.

### Python Script (Direct Invocation Example)

You can also invoke the agent directly from a Python script. The `agent_runnable` object is created with a default configuration.

```python
from my_agent.agent import agent_runnable # agent_runnable is compiled with default config
from my_agent.utils.state import AgentState
from my_agent.utils.config import AgentConfiguration # To potentially use a non-default config
from langchain_core.messages import HumanMessage

# If you want to use a non-default configuration:
# from my_agent.agent import create_graph_runnable
# custom_config = AgentConfiguration(llm_temperature_synthesis=0.9)
# custom_agent_runnable = create_graph_runnable(custom_config)
# runnable_to_use = custom_agent_runnable

runnable_to_use = agent_runnable # Using the default configured one

# Prepare the initial state (input)
# Ensure all fields of AgentState are present if not using default_factory in TypedDict
initial_input = AgentState(
    user_query="Hello, what can you do?",
    chat_history=[],
    # Fill other AgentState fields with initial empty/default values
    query_type="",
    plan=[],
    current_step_index=0,
    retrieved_documents=[],
    file_contents={},
    final_answer="",
    missing_info_request="",
    intermediate_steps=[]
)

print("Streaming agent execution events:")
for event in runnable_to_use.stream(initial_input, {"recursion_limit": 100}):
    print(f"--- Event: {event.get('event', 'data')} ---")
    # Event will be a dictionary, print keys to see content
    for key, value in event.items():
        if key != 'messages': # messages can be very verbose
             print(f"  {key}: {value}")
    print("--- End Event ---")

# The final answer is typically in the last event for the 'response_synthesizer' node
# or under the '__end__' key if using invoke.
# For stream, you'd collect messages or look at the state from the last relevant event.

# Example of invoking and getting the final state (less verbose for some cases)
# final_state = runnable_to_use.invoke(initial_input, {"recursion_limit": 100})
# print("\nFinal Answer from invoke:")
# print(final_state.get('final_answer'))
```

## Running Tests

To run the automated tests for tools, nodes, and the agent:

1.  Ensure your `.env` file is set up with a valid `GOOGLE_API_KEY`, as some tests make live API calls (these tests will be skipped if the key is not present).
2.  Navigate to the `my-app` directory in your terminal.
3.  Run the unittest discovery:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```

## Configuration

The agent's behavior can be customized via the `AgentConfiguration` class located in `my_app/my_agent/utils/config.py`. This includes settings such as:

-   LLM model name (`llm_model_name`)
-   Embedding model name (`embedding_model_name`)
-   LLM temperatures for default tasks and synthesis (`llm_temperature_default`, `llm_temperature_synthesis`)
-   Retriever's K value (`retriever_k`)
-   Maximum sub-queries for the retriever (`max_sub_queries`)

To change the default behavior for the globally available `agent_runnable` in `my_agent/agent.py`, you would modify the `default_config` instance created in that file before `agent_runnable` is compiled, or modify the default values directly in `AgentConfiguration`. For more dynamic configuration, you can use the `create_graph_runnable(config: AgentConfiguration)` factory function also available in `my_agent/agent.py` to create differently configured agent instances.
