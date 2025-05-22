import os
import json
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Attempt to import AgentState, trying different paths
try:
    from ..utils.state import AgentState  # For use within the package
except (ImportError, ValueError):
    try:
        from utils.state import AgentState # For local testing if utils is in PYTHONPATH
    except ImportError:
        from my_agent.utils.state import AgentState # Fallback for specific execution context


# Load environment variables from .env file, useful for local testing
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY not found in environment variables. Please set it in the .env file for the LLM to work.")

# 1. LLM Initialization
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# 2. Output Parser Definition (Pydantic Model)
class QueryAnalysis(BaseModel):
    """
    Defines the structured output for the query analysis.
    """
    query_type: str = Field(
        description="Classification of the user query. Must be one of: 'greeting', 'ambiguous', 'retrieval', 'file_read', 'info_response', 'other'"
    )
    extracted_file_paths: list[str] = Field(
        default_factory=list,
        description="List of file paths extracted if query_type is 'file_read'."
    )
    missing_info_question: str = Field(
        default="",
        description="A question to ask the user if query_type is 'ambiguous' and more information is needed."
    )
    reasoning: str = Field(
        description="Brief explanation of why the query was classified as such."
    )

# 3. Query Analyzer Function
def analyze_query_node(state: AgentState) -> AgentState:
    """
    Analyzes the user query to determine its type and extracts relevant information.
    Updates the agent state with the analysis results.
    """
    user_query = state['user_query']
    chat_history = state.get('chat_history', []) # Ensure chat_history exists

    # Prepare chat history for the prompt (simple string representation)
    formatted_chat_history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in chat_history]
    )

    prompt_template = """
    You are an expert query analyzer. Your task is to understand the user's query in the context of any ongoing conversation and classify it.
    Respond using the JSON format that matches the Pydantic model 'QueryAnalysis'.

    User Query: {user_query}
    Chat History:
    {chat_history}

    Based on this, classify the query and provide the requested information.
    The available query types are:
    - 'greeting': For general greetings or chitchat (e.g., "hello", "how are you?").
    - 'ambiguous': If the query is unclear, vague, or lacks necessary details to be actionable (e.g., "tell me about it", "what can you do?").
    - 'retrieval': If the query asks for information that can likely be found in a knowledge base or requires general knowledge (e.g., "what is LangGraph?", "explain quantum physics").
    - 'file_read': If the query explicitly asks to read, describe, summarize, or process specific files (e.g., "read file.txt", "summarize document.pdf", "what's in /path/to/my/code.py?").
    - 'info_response': If the user is providing an answer or information in response to a previous question from you (especially if you asked for clarification).
    - 'other': If the query doesn't fit any of the above, is out of scope, or is a command you cannot execute (e.g., "play music", "send an email").

    Instructions for specific fields:
    - 'query_type': (string, required) Choose one from the list above.
    - 'extracted_file_paths': (list of strings, optional) If query_type is 'file_read', extract any potential file paths mentioned. File paths can be relative or absolute. Examples: "file.txt", "./docs/report.pdf", "/usr/local/data.csv".
    - 'missing_info_question': (string, optional) If query_type is 'ambiguous', formulate a concise question to ask the user to clarify their request. Otherwise, leave empty.
    - 'reasoning': (string, required) Provide a brief explanation for your classification and other decisions (e.g., why it's ambiguous, how file paths were identified).

    Ensure your entire response is a single JSON object that conforms to the QueryAnalysis model.
    """

    analyzer_prompt = ChatPromptTemplate.from_template(template=prompt_template)

    # Chain the prompt with the LLM and the Pydantic output parser
    # The .with_structured_output method automatically handles the JSON conversion
    # and parsing into the Pydantic model.
    query_analyzer_chain = analyzer_prompt | llm.with_structured_output(QueryAnalysis)

    try:
        analysis_result: QueryAnalysis = query_analyzer_chain.invoke({
            "user_query": user_query,
            "chat_history": formatted_chat_history
        })

        # Update state
        state['query_type'] = analysis_result.query_type
        state['missing_info_request'] = analysis_result.missing_info_question
        
        if analysis_result.extracted_file_paths:
            # Initialize file_contents dict if paths are found, preserving existing entries if any
            if 'file_contents' not in state or not isinstance(state['file_contents'], dict):
                state['file_contents'] = {}
            for path in analysis_result.extracted_file_paths:
                if path not in state['file_contents']: # Avoid overwriting if already read
                    state['file_contents'][path] = "" 
        
        # Optionally store reasoning (e.g., in intermediate_steps)
        # For now, just printing for debugging during development
        print(f"Query Analysis Reasoning: {analysis_result.reasoning}")
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("QueryAnalysis", analysis_result.dict()))


    except Exception as e:
        print(f"Error during query analysis: {e}")
        # Fallback state in case of error
        state['query_type'] = "error"
        state['missing_info_request'] = "I encountered an error trying to understand your request. Could you please rephrase or simplify it?"
        state['file_contents'] = {}
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("QueryAnalysisError", str(e)))


    return state

if __name__ == '__main__':
    # This is a basic test, assuming GOOGLE_API_KEY is set in .env
    # For more comprehensive testing, you'd mock AgentState and BaseMessage
    print("Testing Query Analyzer Node...")

    # Mock AgentState for testing
    class MockMessage:
        def __init__(self, type_str, content_str):
            self.type = type_str
            self.content = content_str

    test_state_greeting: AgentState = {
        "user_query": "Hello there!",
        "chat_history": [],
        "query_type": "",
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "",
        "intermediate_steps": []
    }
    print("\n--- Test Case: Greeting ---")
    updated_state_greeting = analyze_query_node(test_state_greeting)
    print(f"User Query: {test_state_greeting['user_query']}")
    print(f"Query Type: {updated_state_greeting['query_type']}")
    print(f"Missing Info Request: {updated_state_greeting['missing_info_request']}")
    print(f"File Contents: {updated_state_greeting.get('file_contents')}")
    # print(f"Intermediate Steps: {updated_state_greeting['intermediate_steps']}")


    test_state_file: AgentState = {
        "user_query": "Can you read the file named report.txt and also check /tmp/data.csv?",
        "chat_history": [],
        "query_type": "",
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "",
        "intermediate_steps": []
    }
    print("\n--- Test Case: File Read ---")
    updated_state_file = analyze_query_node(test_state_file)
    print(f"User Query: {test_state_file['user_query']}")
    print(f"Query Type: {updated_state_file['query_type']}")
    print(f"Extracted Paths: {updated_state_file.get('file_contents', {}).keys()}") # Check keys of file_contents
    print(f"Missing Info Request: {updated_state_file['missing_info_request']}")
    # print(f"Intermediate Steps: {updated_state_file['intermediate_steps']}")

    test_state_ambiguous: AgentState = {
        "user_query": "Tell me more.",
        "chat_history": [MockMessage(type_str="ai", content_str="I can help with information retrieval and file processing.")],
        "query_type": "",
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "",
        "intermediate_steps": []
    }
    print("\n--- Test Case: Ambiguous ---")
    updated_state_ambiguous = analyze_query_node(test_state_ambiguous)
    print(f"User Query: {test_state_ambiguous['user_query']}")
    print(f"Chat History: {[(msg.type, msg.content) for msg in test_state_ambiguous['chat_history']]}")
    print(f"Query Type: {updated_state_ambiguous['query_type']}")
    print(f"Missing Info Request: {updated_state_ambiguous['missing_info_request']}")
    # print(f"Intermediate Steps: {updated_state_ambiguous['intermediate_steps']}")

    test_state_retrieval: AgentState = {
        "user_query": "What is the capital of France?",
        "chat_history": [],
        "query_type": "",
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "",
        "intermediate_steps": []
    }
    print("\n--- Test Case: Retrieval ---")
    updated_state_retrieval = analyze_query_node(test_state_retrieval)
    print(f"User Query: {test_state_retrieval['user_query']}")
    print(f"Query Type: {updated_state_retrieval['query_type']}")
    print(f"Missing Info Request: {updated_state_retrieval['missing_info_request']}")
    # print(f"Intermediate Steps: {updated_state_retrieval['intermediate_steps']}")

    test_state_info_response: AgentState = {
        "user_query": "Yes, I meant the main configuration file for the web server.",
        "chat_history": [
            MockMessage(type_str="human", content_str="Can you check the config?"),
            MockMessage(type_str="ai", content_str="Which configuration file are you referring to?")
        ],
        "query_type": "", # This would be set by previous turn in a real scenario
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "Which configuration file are you referring to?", # From previous turn
        "intermediate_steps": []
    }
    print("\n--- Test Case: Info Response ---")
    updated_state_info_response = analyze_query_node(test_state_info_response)
    print(f"User Query: {test_state_info_response['user_query']}")
    print(f"Chat History: {[(msg.type, msg.content) for msg in test_state_info_response['chat_history']]}")
    print(f"Query Type: {updated_state_info_response['query_type']}")
    print(f"Missing Info Request: {updated_state_info_response['missing_info_request']}") # Should ideally be cleared or updated
    # print(f"Intermediate Steps: {updated_state_info_response['intermediate_steps']}")

    print("\nQuery Analyzer Node testing complete.")


# --- Planner Node ---

# 4. Output Parser Definition for Planner (Pydantic Model)
class Plan(BaseModel):
    """
    Defines the structured output for the planning phase.
    """
    steps: list[str] = Field(
        description="A list of actionable steps to address the user's query. Each step should clearly state the action (e.g., 'Use retriever for X', 'Read file Y', 'Synthesize answer based on gathered information')."
    )
    reasoning: str = Field(
        description="Brief explanation of why this plan was generated."
    )

# 5. Planner Function
def planner_node(state: AgentState) -> AgentState:
    """
    Generates a plan of action based on the analyzed query and available information.
    Updates the agent state with the plan.
    """
    user_query = state['user_query']
    query_type = state['query_type']
    retrieved_documents = state.get('retrieved_documents', [])
    file_contents = state.get('file_contents', {})
    chat_history = state.get('chat_history', [])

    # Only plan for query types that require further action
    if query_type not in ['retrieval', 'file_read', 'info_response', 'complex_qa']: # Added complex_qa if needed
        print(f"Planner Node: Skipping planning for query_type '{query_type}'.")
        # Optionally clear old plan
        state['plan'] = []
        state['current_step_index'] = 0
        return state

    # Prepare context for LLM
    formatted_chat_history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in chat_history]
    )
    document_summaries = [doc.page_content[:100] + "..." for doc in retrieved_documents] # Simple summarization
    available_files = list(file_contents.keys())

    planner_prompt_template = """
    You are an expert planner. Your task is to create a step-by-step plan to answer the user's query using the available tools and information.
    Respond using the JSON format that matches the Pydantic model 'Plan'.

    User Query: {user_query}
    Query Type: {query_type}
    Chat History:
    {chat_history}

    Available Tools:
    - FaissRetrieverTool: Retrieves relevant documents. Use like "Use retriever for 'search terms'". This is for general knowledge or finding information in a large dataset.
    - FileReadTool: Reads the content of a specified file. Use like "Read file 'path/to/file.txt'". This is for when the user explicitly asks to read or process a file.

    Current Knowledge:
    - Retrieved Documents (summaries): {document_summaries}
    - Available File Contents (filenames only): {available_files}

    Based on the user query, its type, and current knowledge, generate a plan.
    The plan should be a list of actionable steps.
    - If the query is 'file_read' and file paths were extracted by the analyzer, the first step(s) should usually be "Read file 'path/to/file.txt'" for each relevant file.
    - If the query is 'retrieval', a step like "Use retriever for 'search terms related to query'" might be needed if current documents are insufficient.
    - If the query is 'info_response', it means the user provided some information. The plan should decide the next action based on this new info (e.g., use retriever, read a file if specified, or synthesize).
    - If existing information (documents or file contents) seems sufficient to answer the query, the plan might just be to synthesize the answer.
    - The final step of any information gathering plan (that involves using retriever or reading files) MUST be "Synthesize answer using all gathered information."
    - If the query type is 'greeting', 'ambiguous', or 'other', and this planner is somehow called, generate an empty plan or a plan with a single step like "Respond directly based on query type." (Though routing should prevent this).

    Instructions for specific fields:
    - 'steps': (list of strings, required) Actionable steps.
    - 'reasoning': (string, required) Brief explanation for this plan.

    Ensure your entire response is a single JSON object that conforms to the Plan model.
    Example for a file read query:
    User Query: "Read my_document.txt and tell me what it says about project X."
    Query Type: "file_read"
    Available File Contents: []
    Plan:
    {{
        "steps": ["Read file 'my_document.txt'", "Synthesize answer using all gathered information."],
        "reasoning": "User asked to read a specific file. The plan is to read it and then synthesize the answer."
    }}

    Example for a retrieval query:
    User Query: "What are the latest advancements in AI?"
    Query Type: "retrieval"
    Retrieved Documents (summaries): ["Old AI paper summary..."]
    Plan:
    {{
        "steps": ["Use retriever for 'latest advancements in AI'", "Synthesize answer using all gathered information."],
        "reasoning": "User asked a general knowledge question. Existing documents seem outdated, so retrieval is needed before synthesizing."
    }}
    """

    planner_prompt = ChatPromptTemplate.from_template(template=planner_prompt_template)
    planner_chain = planner_prompt | llm.with_structured_output(Plan)

    try:
        plan_result: Plan = planner_chain.invoke({
            "user_query": user_query,
            "query_type": query_type,
            "chat_history": formatted_chat_history,
            "document_summaries": document_summaries,
            "available_files": available_files
        })

        state['plan'] = plan_result.steps
        state['current_step_index'] = 0
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("PlannerNode", plan_result.dict()))
        print(f"Planner Node Reasoning: {plan_result.reasoning}")

    except Exception as e:
        print(f"Error during planning: {e}")
        state['plan'] = ["Error: Could not generate a plan. Please try rephrasing your query."]
        state['current_step_index'] = 0
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("PlannerNodeError", str(e)))

    return state


if __name__ == '__main__':
    # This is a basic test, assuming GOOGLE_API_KEY is set in .env
    # For more comprehensive testing, you'd mock AgentState and BaseMessage
    print("Testing Query Analyzer Node...")

    # Mock AgentState for testing
    class MockMessage:
        def __init__(self, type_str, content_str):
            self.type = type_str
            self.content = content_str

    # ... (previous test cases for analyze_query_node remain) ...
    test_state_greeting: AgentState = {
        "user_query": "Hello there!",
        "chat_history": [],
        "query_type": "",
        "plan": [],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "",
        "missing_info_request": "",
        "intermediate_steps": []
    }
    print("\n--- Test Case: Greeting (Query Analyzer) ---")
    updated_state_greeting = analyze_query_node(test_state_greeting)
    print(f"  User Query: {test_state_greeting['user_query']}")
    print(f"  Query Type: {updated_state_greeting['query_type']}")
    # Test planner with greeting type
    print("--- Test Case: Greeting (Planner) ---")
    planned_state_greeting = planner_node(updated_state_greeting)
    print(f"  Plan: {planned_state_greeting['plan']}")


    test_state_file: AgentState = {
        "user_query": "Can you read the file named report.txt and also check /tmp/data.csv?",
        "chat_history": [],
        "query_type": "", # To be filled by analyzer
        "plan": [], "current_step_index": 0, "retrieved_documents": [],
        "file_contents": {}, # To be filled by analyzer
        "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: File Read (Query Analyzer) ---")
    updated_state_file = analyze_query_node(test_state_file)
    print(f"  User Query: {test_state_file['user_query']}")
    print(f"  Query Type: {updated_state_file['query_type']}")
    print(f"  Extracted Paths for file_contents: {updated_state_file.get('file_contents', {}).keys()}")
    # Test planner with file_read type
    print("--- Test Case: File Read (Planner) ---")
    planned_state_file = planner_node(updated_state_file)
    print(f"  Plan: {planned_state_file['plan']}")
    if planned_state_file['intermediate_steps']:
        for step in planned_state_file['intermediate_steps']:
            if step[0] == "PlannerNode":
                print(f"  Planner Reasoning: {step[1]['reasoning']}")


    test_state_ambiguous: AgentState = {
        "user_query": "Tell me more.",
        "chat_history": [MockMessage(type_str="ai", content_str="I can help with information retrieval and file processing.")],
        "query_type": "", "plan": [], "current_step_index": 0, "retrieved_documents": [],
        "file_contents": {}, "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: Ambiguous (Query Analyzer) ---")
    updated_state_ambiguous = analyze_query_node(test_state_ambiguous)
    print(f"  User Query: {test_state_ambiguous['user_query']}")
    print(f"  Query Type: {updated_state_ambiguous['query_type']}")
    print(f"  Missing Info: {updated_state_ambiguous['missing_info_request']}")
    # Test planner with ambiguous type
    print("--- Test Case: Ambiguous (Planner) ---")
    planned_state_ambiguous = planner_node(updated_state_ambiguous)
    print(f"  Plan: {planned_state_ambiguous['plan']}")


    test_state_retrieval: AgentState = {
        "user_query": "What is the LangGraph library?",
        "chat_history": [],
        "query_type": "", "plan": [], "current_step_index": 0, "retrieved_documents": [],
        "file_contents": {}, "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: Retrieval (Query Analyzer) ---")
    updated_state_retrieval = analyze_query_node(test_state_retrieval)
    print(f"  User Query: {test_state_retrieval['user_query']}")
    print(f"  Query Type: {updated_state_retrieval['query_type']}")
    # Test planner with retrieval type
    print("--- Test Case: Retrieval (Planner) ---")
    planned_state_retrieval = planner_node(updated_state_retrieval)
    print(f"  Plan: {planned_state_retrieval['plan']}")
    if planned_state_retrieval['intermediate_steps']:
        for step in planned_state_retrieval['intermediate_steps']:
            if step[0] == "PlannerNode":
                print(f"  Planner Reasoning: {step[1]['reasoning']}")


    test_state_info_response_for_plan: AgentState = {
        "user_query": "Yes, the file is /my/data/app.log",
        "chat_history": [
            MockMessage(type_str="human", content_str="Can you process the log?"),
            MockMessage(type_str="ai", content_str="Sure, which log file are you referring to?")
        ],
        "query_type": "info_response", # Set directly for this test case
        "plan": [], "current_step_index": 0, "retrieved_documents": [],
        "file_contents": {"/my/data/app.log": ""}, # Assume analyzer picked this up if it re-ran
        "final_answer": "", "missing_info_request": "", "intermediate_steps": [
            ("QueryAnalysis", {"query_type": "info_response", "extracted_file_paths": ["/my/data/app.log"], "missing_info_question": "", "reasoning": "User provided a file path in response to a question."})
        ]
    }
    print("\n--- Test Case: Info Response (Planner) ---")
    # Directly call planner as query_type is already 'info_response'
    planned_state_info_response = planner_node(test_state_info_response_for_plan)
    print(f"  User Query: {test_state_info_response_for_plan['user_query']}")
    print(f"  Query Type: {planned_state_info_response['query_type']}")
    print(f"  File Contents: {planned_state_info_response['file_contents']}")
    print(f"  Plan: {planned_state_info_response['plan']}")
    if planned_state_info_response['intermediate_steps']:
        for step_entry in planned_state_info_response['intermediate_steps']:
            if step_entry[0] == "PlannerNode":
                print(f"  Planner Reasoning: {step_entry[1]['reasoning']}")


    print("\nQuery Analyzer and Planner Node testing complete.")


# --- Tool Executor Node ---

# Attempt to import tools, trying different paths
try:
    from ..utils.tools import faiss_retriever_tool, file_read_tool # For use within the package
    from langchain_core.tools import Tool # Already imported but good to be explicit
    from langchain_core.documents import Document # For type hinting
    import typing # For List, Dict
except (ImportError, ValueError):
    try:
        from utils.tools import faiss_retriever_tool, file_read_tool # For local testing
        from langchain_core.tools import Tool
        from langchain_core.documents import Document
        import typing
    except ImportError:
        # This case might occur if running a script from the root of my-app directly
        # and utils/tools.py are not in python path in that specific context.
        # For robust execution, ensure PYTHONPATH is set correctly or the package structure is handled.
        print("Warning: Could not import tools directly, attempting my_agent.utils.tools")
        from my_agent.utils.tools import faiss_retriever_tool, file_read_tool
        from langchain_core.tools import Tool
        from langchain_core.documents import Document
        import typing


def tool_executor_node(state: AgentState, tools: typing.List[Tool]) -> AgentState:
    """
    Executes the tool specified in the current step of the plan.
    Updates the agent state with the tool's output and increments the step index.
    """
    plan = state.get('plan', [])
    current_step_index = state.get('current_step_index', 0)

    if not plan or current_step_index >= len(plan):
        print("Tool Executor Node: No plan or plan execution finished.")
        # No action if plan is empty or finished
        return state

    current_step: str = plan[current_step_index]
    tool_output = None
    tool_to_execute = None
    tool_input = None

    # Ensure intermediate_steps is initialized
    if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
        state['intermediate_steps'] = []
    if 'retrieved_documents' not in state or not isinstance(state['retrieved_documents'], list):
        state['retrieved_documents'] = []
    if 'file_contents' not in state or not isinstance(state['file_contents'], dict):
        state['file_contents'] = {}

    print(f"Tool Executor Node: Executing step {current_step_index + 1}/{len(plan)}: '{current_step}'")

    # 1. Parse the current_step to identify tool and input
    # Using regex to extract quoted arguments
    retriever_match = re.search(r"Use retriever for ['\"](.*?)['\"]", current_step, re.IGNORECASE)
    file_read_match = re.search(r"Read file ['\"](.*?)['\"]", current_step, re.IGNORECASE)

    if retriever_match:
        query = retriever_match.group(1)
        tool_to_execute = next((t for t in tools if t.name == "FaissRetrieverTool"), None)
        tool_input = query # faiss_retriever_tool.func (retrieve_documents) expects a string query
        action_type = "FaissRetrieverTool"
    elif file_read_match:
        file_path = file_read_match.group(1)
        tool_to_execute = next((t for t in tools if t.name == "FileReadTool"), None)
        tool_input = file_path # file_read_tool.func (read_file_content) expects a string file_path
        action_type = "FileReadTool"
    elif current_step.lower().startswith("synthesize answer"):
        print("Tool Executor Node: Reached synthesis step. No tool execution needed here.")
        state['current_step_index'] = current_step_index + 1
        state['intermediate_steps'].append((current_step, "No tool execution, proceeding to synthesis."))
        return state
    else:
        error_message = f"Tool Executor Node: Could not parse step or identify tool: '{current_step}'"
        print(error_message)
        state['intermediate_steps'].append((current_step, error_message))
        state['current_step_index'] = current_step_index + 1 # Move to next step to avoid loop
        return state

    # 2. Execute the tool
    if tool_to_execute and tool_input is not None:
        try:
            print(f"  Invoking tool: {tool_to_execute.name} with input: '{tool_input}'")
            tool_output = tool_to_execute.invoke(tool_input)
            print(f"  Tool Output ({tool_to_execute.name}): {str(tool_output)[:200]}...") # Print snippet of output

            # 3. Update State based on tool
            if action_type == "FaissRetrieverTool":
                # Ensure retrieved_documents is a list of Documents
                # The tool's func `retrieve_documents` returns List[Document]
                if isinstance(tool_output, list) and all(isinstance(doc, Document) for doc in tool_output):
                    # Add new documents, avoiding duplicates by page_content
                    existing_doc_contents = {doc.page_content for doc in state['retrieved_documents']}
                    for doc in tool_output:
                        if doc.page_content not in existing_doc_contents:
                            state['retrieved_documents'].append(doc)
                            existing_doc_contents.add(doc.page_content)
                else:
                    print(f"Warning: FaissRetrieverTool output was not a list of Documents: {type(tool_output)}")
                    state['intermediate_steps'].append((current_step, f"Warning: Tool output not List[Document]: {tool_output}"))


            elif action_type == "FileReadTool":
                # The tool's func `read_file_content` returns a string (content or error message)
                # The input was file_path
                if isinstance(tool_output, str):
                    state['file_contents'][tool_input] = tool_output # tool_input here is the file_path
                else:
                    print(f"Warning: FileReadTool output was not a string: {type(tool_output)}")
                    state['intermediate_steps'].append((current_step, f"Warning: Tool output not str: {tool_output}"))
            
            state['intermediate_steps'].append((current_step, tool_output))

        except Exception as e:
            error_message = f"Tool Executor Node: Error executing tool {tool_to_execute.name}: {e}"
            print(error_message)
            state['intermediate_steps'].append((current_step, error_message))
    elif not tool_to_execute:
        error_message = f"Tool Executor Node: Tool for step '{current_step}' not found in provided tools list."
        print(error_message)
        state['intermediate_steps'].append((current_step, error_message))
    
    # 4. Increment current_step_index
    state['current_step_index'] = current_step_index + 1
    return state


if __name__ == '__main__':
    # This is a basic test, assuming GOOGLE_API_KEY is set in .env
    # For more comprehensive testing, you'd mock AgentState and BaseMessage
    print("Testing Query Analyzer Node...")

    # Mock AgentState for testing
    class MockMessage:
        def __init__(self, type_str, content_str):
            self.type = type_str
            self.content = content_str

    # ... (previous test cases for analyze_query_node and planner_node remain) ...
    # Re-define them here for clarity and to avoid scrolling up/down during diffs
    test_state_greeting: AgentState = {
        "user_query": "Hello there!", "chat_history": [], "query_type": "", "plan": [],
        "current_step_index": 0, "retrieved_documents": [], "file_contents": {},
        "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: Greeting (Query Analyzer) ---")
    updated_state_greeting = analyze_query_node(test_state_greeting)
    print(f"  User Query: {test_state_greeting['user_query']}")
    print(f"  Query Type: {updated_state_greeting['query_type']}")
    print("--- Test Case: Greeting (Planner) ---")
    planned_state_greeting = planner_node(updated_state_greeting)
    print(f"  Plan: {planned_state_greeting['plan']}")
    print("--- Test Case: Greeting (Tool Executor) ---") # Should do nothing as plan is empty or not tool-related
    executed_state_greeting = tool_executor_node(planned_state_greeting, [faiss_retriever_tool, file_read_tool])
    print(f"  Intermediate Steps after Tool Executor: {executed_state_greeting['intermediate_steps']}")


    # Test case for Tool Executor with FileReadTool
    # Setup a dummy file for the FileReadTool to use
    dummy_file_name = "test_executor_dummy_file.txt"
    with open(dummy_file_name, "w") as f:
        f.write("This is content from the executor test dummy file.")

    test_state_file_exec: AgentState = {
        "user_query": f"Read the file {dummy_file_name}",
        "chat_history": [],
        "query_type": "file_read", # Assume analyzer set this
        "plan": [f"Read file '{dummy_file_name}'", "Synthesize answer using all gathered information."],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {dummy_file_name: ""}, # Analyzer might have added the key
        "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: File Read (Tool Executor) ---")
    print(f"  Initial Plan: {test_state_file_exec['plan']}")
    print(f"  Initial File Contents: {test_state_file_exec['file_contents']}")
    executed_state_file = tool_executor_node(test_state_file_exec, [faiss_retriever_tool, file_read_tool])
    print(f"  File Contents after execution: {executed_state_file['file_contents'].get(dummy_file_name)}")
    print(f"  Current Step Index: {executed_state_file['current_step_index']}")
    assert executed_state_file['file_contents'].get(dummy_file_name) == "This is content from the executor test dummy file."
    assert executed_state_file['current_step_index'] == 1
    # Clean up dummy file
    if os.path.exists(dummy_file_name):
        os.remove(dummy_file_name)

    # Test case for Tool Executor with FaissRetrieverTool
    # This requires GOOGLE_API_KEY for the embeddings and LLM used by the tool.
    # The dummy documents for FAISS are in tools.py itself.
    test_state_retrieval_exec: AgentState = {
        "user_query": "What is LangGraph?",
        "chat_history": [],
        "query_type": "retrieval", # Assume analyzer set this
        "plan": ["Use retriever for 'LangGraph library features'", "Synthesize answer using all gathered information."],
        "current_step_index": 0,
        "retrieved_documents": [],
        "file_contents": {},
        "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: Retrieval (Tool Executor) ---")
    print(f"  Initial Plan: {test_state_retrieval_exec['plan']}")
    executed_state_retrieval = tool_executor_node(test_state_retrieval_exec, [faiss_retriever_tool, file_read_tool])
    print(f"  Retrieved Documents after execution: {len(executed_state_retrieval['retrieved_documents'])} documents.")
    for i, doc in enumerate(executed_state_retrieval['retrieved_documents']):
        print(f"    Doc {i+1}: {doc.page_content[:100]}...")
    print(f"  Current Step Index: {executed_state_retrieval['current_step_index']}")
    assert len(executed_state_retrieval['retrieved_documents']) > 0 # Expecting at least one relevant doc
    assert executed_state_retrieval['current_step_index'] == 1
    
    # Test for 'Synthesize answer' step (should not execute any tool)
    test_state_synthesize_step: AgentState = {
        "user_query": "N/A",
        "chat_history": [],
        "query_type": "N/A",
        "plan": ["Synthesize answer using all gathered information."],
        "current_step_index": 0,
        "retrieved_documents": [Document(page_content="Some info")],
        "file_contents": {"some_file.txt": "some content"},
        "final_answer": "", "missing_info_request": "", "intermediate_steps": []
    }
    print("\n--- Test Case: Synthesize Step (Tool Executor) ---")
    executed_state_synth = tool_executor_node(test_state_synthesize_step, [faiss_retriever_tool, file_read_tool])
    print(f"  Intermediate Steps: {executed_state_synth['intermediate_steps']}")
    print(f"  Current Step Index: {executed_state_synth['current_step_index']}")
    assert executed_state_synth['current_step_index'] == 1
    assert "No tool execution" in executed_state_synth['intermediate_steps'][0][1]


    print("\nQuery Analyzer, Planner, and Tool Executor Node testing complete.")


# --- Response Synthesizer Node ---

# LLM for Synthesis (can have different temperature)
# The main 'llm' is temperature 0. This one can be slightly higher for more natural response.
# However, for strict adherence to context, temperature 0 might still be preferred.
# For now, let's create a new instance for synthesis if we want different settings,
# or we can reuse the existing 'llm' if temperature 0 is fine.
# Re-using existing 'llm' for now to keep it simple, but this is where you might change it.
# llm_synthesis = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
# Using the same llm as other nodes for now.
llm_synthesis = llm # Or initialize a new one if different settings are needed.


def response_synthesizer_node(state: AgentState) -> AgentState:
    """
    Generates a final response to the user based on the query type,
    gathered information (documents, file contents), and chat history.
    """
    user_query = state['user_query']
    query_type = state['query_type']
    retrieved_documents = state.get('retrieved_documents', [])
    file_contents = state.get('file_contents', {})
    chat_history = state.get('chat_history', [])
    missing_info_request = state.get('missing_info_request', "") # From analyzer

    final_answer = ""

    if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
        state['intermediate_steps'] = []

    print(f"Response Synthesizer Node: Generating response for query_type '{query_type}'.")

    if query_type == 'greeting':
        final_answer = "Hello! I am a RAG agent. I can help you by retrieving information from documents and files. How can I assist you today?"
    elif query_type == 'ambiguous' and missing_info_request:
        final_answer = missing_info_request
    elif query_type == 'error':
        final_answer = missing_info_request or "I encountered an issue processing your request. Please try rephrasing your query or try again later."
    elif query_type == 'other':
        final_answer = "I'm designed to help with information retrieval and file processing. I'm not sure how to help with that. Could you ask something related to those topics?"
    else: # 'retrieval', 'file_read', 'info_response' (after processing), or other synthesis-requiring types
        # Prepare context for LLM
        formatted_chat_history = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in chat_history]
        )
        document_context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_documents)])
        
        file_context_parts = []
        for path, content in file_contents.items():
            if "Error: File not found" in content or "An unexpected error occurred" in content:
                 file_context_parts.append(f"Note: Could not read file '{path}': {content}")
            else:
                file_context_parts.append(f"Content of file '{path}':\n{content}")
        file_context = "\n\n---\n\n".join(file_context_parts)


        synthesis_prompt_template = """
        You are a helpful AI assistant. Your primary task is to answer the user's query based *strictly* and *only* on the provided context from chat history, retrieved documents, and file contents.
        Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.
        If the provided context is insufficient to answer the query, clearly state that you cannot answer based on the given information.
        When information is available, synthesize it into a coherent, helpful, and comprehensive answer.
        If you use information from specific documents or files, please cite them (e.g., "According to Document 1...", "Based on the content of file 'example.txt'...", "Document 2 and file 'report.pdf' suggest...").
        If multiple documents or files address the query, try to synthesize the information, noting any agreements or disagreements if applicable.

        User Query: {user_query}

        Previous Chat History (for context, but prioritize current query and new information):
        ---
        {chat_history}
        ---

        Retrieved Document Context:
        ---
        {document_context}
        ---

        File Content Context:
        ---
        {file_context}
        ---

        Based *solely* on the information detailed in the 'Retrieved Document Context', 'File Content Context', and relevant 'Chat History', provide a comprehensive answer to the 'User Query'.
        If no relevant information is found in the context to answer the query, explicitly say so. For example: "Based on the provided documents and files, I cannot answer your query about X."
        Do not add any information that is not present in the provided context.
        """
        
        synthesis_prompt = ChatPromptTemplate.from_template(template=synthesis_prompt_template)
        # Using the main llm instance here. If a different temperature is desired for synthesis,
        # initialize and use llm_synthesis = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
        synthesis_chain = synthesis_prompt | llm_synthesis 

        try:
            print("  Synthesizing response from context...")
            # print(f"    Document Context: {document_context[:300]}...")
            # print(f"    File Context: {file_context[:300]}...")
            
            llm_response = synthesis_chain.invoke({
                "user_query": user_query,
                "chat_history": formatted_chat_history,
                "document_context": document_context or "No documents were retrieved.",
                "file_context": file_context or "No files were read or their contents are not available."
            })
            final_answer = llm_response.content
            print(f"  LLM Synthesis Output: {final_answer[:200]}...")
        except Exception as e:
            print(f"Error during response synthesis: {e}")
            final_answer = "I encountered an error while trying to generate a response based on the available information. Please try again."
            state['intermediate_steps'].append(("ResponseSynthesisError", str(e)))

    state['final_answer'] = final_answer
    state['intermediate_steps'].append(("FinalAnswerGenerated", final_answer))
    return state
