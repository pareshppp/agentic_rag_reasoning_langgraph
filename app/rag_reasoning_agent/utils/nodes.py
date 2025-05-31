import os
import json
import re
import typing # For List, Dict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from langchain_core.documents import Document


# Attempt to import AgentState and AgentConfiguration
try:
    from ..utils.state import AgentState  # For use within the package
    from ..utils.config import AgentConfiguration
except (ImportError, ValueError):
    try:
        from utils.state import AgentState # For local testing if utils is in PYTHONPATH
        from utils.config import AgentConfiguration
    except ImportError:
        from rag_reasoning_agent.utils.state import AgentState # Fallback for specific execution context
        from rag_reasoning_agent.utils.config import AgentConfiguration


# Load environment variables from .env file, useful for local testing
load_dotenv()

# Ensure GOOGLE_API_KEY is set (can remain at module level for early check)
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY not found in environment variables. LLM calls will likely fail.")

# Removed module-level llm instance. LLMs will be initialized within node functions.

# --- Query Analyzer Node ---
class QueryAnalysis(BaseModel):
    """Defines the structured output for the query analysis."""
    query_type: str = Field(description="Classification of the user query. Must be one of: 'greeting', 'ambiguous', 'retrieval', 'file_read', 'info_response', 'other'")
    extracted_file_paths: list[str] = Field(default_factory=list, description="List of file paths extracted if query_type is 'file_read'.")
    missing_info_question: str = Field(default="", description="A question to ask the user if query_type is 'ambiguous' and more information is needed.")
    reasoning: str = Field(description="Brief explanation of why the query was classified as such.")

def analyze_query_node(config: AgentConfiguration, state: AgentState) -> AgentState:
    """Analyzes the user query using configuration."""
    print(f"Node: analyze_query_node (Model: {config.llm_model_name}, Temp: {config.llm_temperature_default})")
    llm = ChatGoogleGenerativeAI(model=config.llm_model_name, temperature=config.llm_temperature_default)
    
    user_query = state['user_query']
    chat_history = state.get('chat_history', [])
    formatted_chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    prompt_template = """
    You are an expert query analyzer... (rest of the prompt remains the same as before)
    User Query: {user_query}
    Chat History:
    {chat_history}
    ... (Ensure your entire response is a single JSON object that conforms to the QueryAnalysis model.)
    """ # Prompt shortened for brevity, it's the same as before.

    analyzer_prompt = ChatPromptTemplate.from_template(template=prompt_template) # Full prompt from previous version
    query_analyzer_chain = analyzer_prompt | llm.with_structured_output(QueryAnalysis)

    try:
        analysis_result: QueryAnalysis = query_analyzer_chain.invoke({
            "user_query": user_query,
            "chat_history": formatted_chat_history
        })
        state['query_type'] = analysis_result.query_type
        state['missing_info_request'] = analysis_result.missing_info_question
        if analysis_result.extracted_file_paths:
            if 'file_contents' not in state or not isinstance(state['file_contents'], dict):
                state['file_contents'] = {}
            for path in analysis_result.extracted_file_paths:
                if path not in state['file_contents']: state['file_contents'][path] = "" 
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("QueryAnalysis", analysis_result.dict()))
        print(f"  Query Analysis Reasoning: {analysis_result.reasoning}")
    except Exception as e:
        print(f"  Error during query analysis: {e}")
        state['query_type'] = "error"
        state['missing_info_request'] = "I encountered an error trying to understand your request."
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("QueryAnalysisError", str(e)))
    return state

# --- Planner Node ---
class Plan(BaseModel):
    """Defines the structured output for the planning phase."""
    steps: list[str] = Field(description="A list of actionable steps. Each step must be a specific tool call string formatted as 'ToolName: \"argument\"' (e.g., \"Use retriever for: \'search query\'\", \"Read file: \'file_path.txt\'\") or the exact string 'Synthesize answer'.")
    reasoning: str = Field(description="Brief explanation of why this plan was generated.")

def planner_node(config: AgentConfiguration, state: AgentState) -> AgentState:
    """Generates a plan of action using configuration."""
    print(f"Node: planner_node (Model: {config.llm_model_name}, Temp: {config.llm_temperature_default})")
    llm = ChatGoogleGenerativeAI(model=config.llm_model_name, temperature=config.llm_temperature_default)

    user_query = state['user_query']
    query_type = state['query_type']
    
    if query_type not in ['retrieval', 'file_read', 'info_response', 'complex_qa']:
        print(f"  Skipping planning for query_type '{query_type}'.")
        state['plan'] = []
        state['current_step_index'] = 0
        return state

    formatted_chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('chat_history', [])])
    document_summaries = [doc.page_content[:100] + "..." for doc in state.get('retrieved_documents', [])]
    available_files = list(state.get('file_contents', {}).keys())

    planner_prompt_template = """
You are an expert planner. Your goal is to create a step-by-step plan to address the user's query based on the query type and available context. Your output *must* be a JSON object conforming to the Plan model below.

Available Tools:
1. FaissRetrieverTool: Use this tool to search for general information or answer questions based on a large knowledge base. To use it, generate a step string: "Use retriever for 'your search query here'"
2. FileReadTool: Use this tool to read the content of a specific file that has been mentioned or is part of the context. To use it, generate a step string: "Read file 'path/to/your/file.txt'"

Plan Steps Formatting:
- Each step in the 'steps' list must be a string.
- If using a tool, the string must exactly match the formats described above (e.g., "Use retriever for 'search query'", "Read file 'file.txt'").
- If you have gathered all necessary information from tool use (or if no tools are needed for the query type) and are ready to formulate the final response, the last step should be the exact string: "Synthesize answer"
- Do not invent new tool names or formats.

Context:
User Query: {user_query}
Query Type: {query_type}
Chat History:
{chat_history}

Current Knowledge:
- Retrieved Documents (summaries): {document_summaries}
- Available File Contents (filenames only): {available_files}

Example Plan for a query like "What is the capital of France and what is in 'info.txt'?":
{{
  "steps": [
    "Use retriever for 'capital of France'",
    "Read file 'info.txt'",
    "Synthesize answer"
  ],
  "reasoning": "First, find the capital of France using the retriever. Second, read the content of info.txt. Finally, synthesize the answer using the gathered information."
}}

Based on the user query and available information, generate the plan.
Ensure your entire response is a single JSON object that conforms to the Plan model.
"""

    planner_prompt = ChatPromptTemplate.from_template(template=planner_prompt_template)
    planner_chain = planner_prompt | llm.with_structured_output(Plan)

    try:
        plan_result: Plan = planner_chain.invoke({
            "user_query": user_query, "query_type": query_type,
            "chat_history": formatted_chat_history,
            "document_summaries": document_summaries,
            "available_files": available_files
        })
        state['plan'] = plan_result.steps
        state['current_step_index'] = 0
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("PlannerNode", plan_result.dict()))
        print(f"  Planner Node Reasoning: {plan_result.reasoning}")
        print(f"  Generated Plan: {plan_result.steps}")
    except Exception as e:
        print(f"  Error during planning: {e}")
        state['plan'] = ["Error: Could not generate a plan."]
        if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list):
            state['intermediate_steps'] = []
        state['intermediate_steps'].append(("PlannerNodeError", str(e)))
    return state

# --- Tool Executor Node ---
def tool_executor_node(config: AgentConfiguration, state: AgentState, tools: typing.List[Tool]) -> AgentState:
    """Executes tools based on the plan; config is for consistency but not directly used by this node's core logic."""
    print(f"Node: tool_executor_node") # Config not directly used by this node's LLM, but good to show it's passed
    plan = state.get('plan', [])
    current_step_index = state.get('current_step_index', 0)

    if not plan or current_step_index >= len(plan):
        print("  No plan or plan execution finished.")
        return state
    
    current_step: str = plan[current_step_index]
    # ... (rest of the logic for parsing step, finding tool, invoking, and updating state remains the same)
    # This logic does not change as it relies on the `tools` list which are now
    # expected to be pre-configured by their factories using AgentConfiguration.
    tool_output = None
    tool_to_execute = None
    tool_input = None

    if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list): state['intermediate_steps'] = []
    if 'retrieved_documents' not in state or not isinstance(state['retrieved_documents'], list): state['retrieved_documents'] = []
    if 'file_contents' not in state or not isinstance(state['file_contents'], dict): state['file_contents'] = {}

    print(f"  Executing step {current_step_index + 1}/{len(plan)}: '{current_step}'")
    retriever_match = re.search(r"Use retriever for ['\"](.*?)['\"]", current_step, re.IGNORECASE)
    file_read_match = re.search(r"Read file ['\"](.*?)['\"]", current_step, re.IGNORECASE)

    action_type = "" # Define action_type to ensure it's always available
    if retriever_match:
        query = retriever_match.group(1)
        tool_to_execute = next((t for t in tools if t.name == "FaissRetrieverTool"), None)
        tool_input = query
        action_type = "FaissRetrieverTool"
    elif file_read_match:
        file_path = file_read_match.group(1)
        tool_to_execute = next((t for t in tools if t.name == "FileReadTool"), None)
        tool_input = file_path
        action_type = "FileReadTool"
    elif current_step.lower().startswith("synthesize answer"):
        print("  Reached synthesis step. No tool execution needed here by executor.")
        state['current_step_index'] = current_step_index + 1
        state['intermediate_steps'].append((current_step, "No tool execution, proceeding to synthesis."))
        return state
    else:
        error_message = f"  Could not parse step or identify tool: '{current_step}'"
        print(error_message)
        state['intermediate_steps'].append((current_step, error_message))
        state['current_step_index'] = current_step_index + 1 
        return state

    if tool_to_execute and tool_input is not None:
        try:
            print(f"    Invoking tool: {tool_to_execute.name} with input: '{tool_input}'")
            tool_output = tool_to_execute.invoke(tool_input) # Tools are now pre-configured
            # print(f"    Tool Output ({tool_to_execute.name}): {str(tool_output)[:200]}...")

            if action_type == "FaissRetrieverTool":
                if isinstance(tool_output, list) and all(isinstance(doc, Document) for doc in tool_output):
                    existing_doc_contents = {doc.page_content for doc in state['retrieved_documents']}
                    for doc in tool_output:
                        if doc.page_content not in existing_doc_contents:
                            state['retrieved_documents'].append(doc)
                            existing_doc_contents.add(doc.page_content)
                else: print(f"Warning: FaissRetrieverTool output was not List[Document]: {type(tool_output)}")
            elif action_type == "FileReadTool":
                if isinstance(tool_output, str): state['file_contents'][tool_input] = tool_output
                else: print(f"Warning: FileReadTool output was not str: {type(tool_output)}")
            state['intermediate_steps'].append((current_step, tool_output))
        except Exception as e:
            error_message = f"    Error executing tool {tool_to_execute.name}: {e}"
            print(error_message)
            state['intermediate_steps'].append((current_step, error_message))
    elif not tool_to_execute:
        error_message = f"    Tool for step '{current_step}' not found."
        print(error_message)
        state['intermediate_steps'].append((current_step, error_message))
    
    state['current_step_index'] = current_step_index + 1
    return state

# --- Response Synthesizer Node ---
def response_synthesizer_node(config: AgentConfiguration, state: AgentState) -> AgentState:
    """Generates a final response using configuration."""
    print(f"Node: response_synthesizer_node (Model: {config.llm_model_name}, Temp: {config.llm_temperature_synthesis})")
    llm_synthesis = ChatGoogleGenerativeAI(model=config.llm_model_name, temperature=config.llm_temperature_synthesis)

    user_query = state['user_query']
    query_type = state['query_type']
    missing_info_request = state.get('missing_info_request', "")
    # ... (rest of variable retrieval from state)
    final_answer = ""
    if 'intermediate_steps' not in state or not isinstance(state['intermediate_steps'], list): state['intermediate_steps'] = []

    if query_type == 'greeting': final_answer = "Hello! I am a RAG agent..."
    elif query_type == 'ambiguous' and missing_info_request: final_answer = missing_info_request
    elif query_type == 'error': final_answer = missing_info_request or "I encountered an issue..."
    elif query_type == 'other': final_answer = "I'm designed to help with information retrieval..."
    else:
        # ... (context preparation: formatted_chat_history, document_context, file_context)
        # Same as before
        formatted_chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('chat_history', [])])
        retrieved_documents = state.get('retrieved_documents', [])
        document_context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_documents)])
        file_contents = state.get('file_contents', {})
        file_context_parts = []
        for path, content in file_contents.items():
            if "Error: File not found" in content or "An unexpected error occurred" in content:
                 file_context_parts.append(f"Note: Could not read file '{path}': {content}")
            else:
                file_context_parts.append(f"Content of file '{path}':\n{content}")
        file_context = "\n\n---\n\n".join(file_context_parts)

        synthesis_prompt_template = """
        You are a helpful AI assistant... (rest of the prompt remains the same as before)
        User Query: {user_query}
        Previous Chat History: ... {chat_history} ...
        Retrieved Document Context: ... {document_context} ...
        File Content Context: ... {file_context} ...
        Based *solely* on the information...
        """ # Prompt shortened for brevity
        
        synthesis_prompt = ChatPromptTemplate.from_template(template=synthesis_prompt_template) # Full prompt
        synthesis_chain = synthesis_prompt | llm_synthesis 

        try:
            print("  Synthesizing response from context...")
            llm_response = synthesis_chain.invoke({
                "user_query": user_query,
                "chat_history": formatted_chat_history,
                "document_context": document_context or "No documents were retrieved.",
                "file_context": file_context or "No files were read or their contents are not available."
            })
            final_answer = llm_response.content
            # print(f"    LLM Synthesis Output: {final_answer[:200]}...")
        except Exception as e:
            print(f"  Error during response synthesis: {e}")
            final_answer = "I encountered an error while trying to generate a response."
            state['intermediate_steps'].append(("ResponseSynthesisError", str(e)))
            
    state['final_answer'] = final_answer
    state['intermediate_steps'].append(("FinalAnswerGenerated", final_answer))
    return state

# No if __name__ == '__main__' block as tests are in test_nodes.py
