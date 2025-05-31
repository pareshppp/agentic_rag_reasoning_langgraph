import os
import json
import typing
from dotenv import load_dotenv
import traceback

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langgraph.graph import END # Import END for conditional routing

# Attempt to import AgentStateV2 and AgentConfigurationV2
try:
    from .state import AgentStateV2
    from .config import AgentConfigurationV2
    from .prompts import RAG_REASONING_SYSTEM_PROMPT # Import the new system prompt
except ImportError:
    # Fallback for testing or direct execution scenarios
    # This is a simplified fallback, direct execution might require sys.path manipulation
    from rag_reasoning_agent_v2.utils.state import AgentStateV2
    from rag_reasoning_agent_v2.utils.config import AgentConfigurationV2
    from rag_reasoning_agent_v2.utils.prompts import RAG_REASONING_SYSTEM_PROMPT

load_dotenv()

# Ensure GOOGLE_API_KEY is set (can remain at module level for early check)
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY not found in environment variables. LLM calls will likely fail.")

def reasoning_node(state: AgentStateV2, agent_config: AgentConfigurationV2, llm_with_tools: ChatGoogleGenerativeAI) -> dict:
    """
    The core reasoning node for the V2 agent.
    It decides whether to answer the user directly or call tools.
    The llm_with_tools is a ChatGoogleGenerativeAI instance already bound with available tools.
    """
    print(f"Node: reasoning_node (Model: {agent_config.llm_model_name}, Temp: {agent_config.llm_temperature_default})")
    
    system_message = SystemMessage(content=RAG_REASONING_SYSTEM_PROMPT)
    messages_from_history = list(state.get('chat_history', []))
    current_user_message = HumanMessage(content=state.get('user_query', ''))
    
    # Construct messages for LLM: System Prompt (new) + Past History + Current User Query
    messages_to_llm = [system_message] + messages_from_history + [current_user_message]

    try:
        ai_response: AIMessage = llm_with_tools.invoke(messages_to_llm)
        print(f"  LLM Response type: {ai_response.type}, content: '{ai_response.content[:100]}...', tool_calls: {ai_response.tool_calls}")

        if ai_response.tool_calls:
            # LLM decided to call tools
            return {
                "intermediate_steps": state.get("intermediate_steps", []) + [ai_response],
                "chat_history": messages_to_llm + [ai_response], # Use full messages_to_llm + response
                # Preserve other relevant state fields
                "user_query": state.get("user_query"), 
                "retrieved_documents": state.get("retrieved_documents", []),
                "file_contents": state.get("file_contents", {}),
                "final_answer": None # No final answer yet as tools are being called
            }
        else:
            # LLM provided a direct answer
            return {
                "final_answer": ai_response.content,
                "chat_history": messages_to_llm + [ai_response], # Use full messages_to_llm + response
                # Preserve other relevant state fields
                "user_query": state.get("user_query"),
                "intermediate_steps": state.get("intermediate_steps", []),
                "retrieved_documents": state.get("retrieved_documents", []),
                "file_contents": state.get("file_contents", {}),
            }

    except Exception as e:
        error_message_str = str(e)
        print(f"  Error invoking LLM in reasoning_node. Exception type: {type(e).__name__}, Message: '{error_message_str}'")
        traceback.print_exc() # Ensure traceback is printed
        
        final_error_content = f"Sorry, I encountered an error while processing your request."
        if error_message_str: # Append specific error if available
            final_error_content += f" Details: {error_message_str}"
        else:
            final_error_content += f" (Exception: {type(e).__name__})"
        
        error_ai_message = AIMessage(content=final_error_content)

        # Preserve original user_query and other relevant parts of the state
        # For chat_history in error, include the system prompt, prior history, current query, and the error message
        updated_chat_history_on_error = (
            [system_message] + 
            list(state.get('chat_history', [])) + 
            [HumanMessage(content=state.get('user_query', ''))] + 
            [error_ai_message]
        )

        return {
            "chat_history": updated_chat_history_on_error,
            "intermediate_steps": state.get("intermediate_steps", []) + [error_ai_message], # Log error as a step
            "final_answer": final_error_content,
            # Preserve other state elements from the input state
            "user_query": state.get("user_query"),
            "retrieved_documents": state.get("retrieved_documents", []),
            "file_contents": state.get("file_contents", {}),
        }


def tool_executor_node_v2(state: AgentStateV2, tools_map: typing.Dict[str, Tool]) -> dict:
    """
    Executes tools based on the tool_calls from the AIMessage in intermediate_steps.
    Appends ToolMessage results to intermediate_steps.
    """
    print("Node: tool_executor_node_v2")
    
    tool_messages_for_this_turn: typing.List[ToolMessage] = []
    updates_to_state: typing.Dict[str, typing.Any] = {}
    current_intermediate_steps = state.get('intermediate_steps', [])
    current_retrieved_documents = state.get('retrieved_documents', [])
    current_file_contents = state.get('file_contents', {}).copy() # Ensure we work with a copy

    last_ai_message: typing.Optional[AIMessage] = None
    if current_intermediate_steps:
        for msg in reversed(current_intermediate_steps):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
    
    if not last_ai_message or not last_ai_message.tool_calls:
        print("  No tool calls to execute from the last AIMessage.")
        if not state.get('final_answer'): # Only set error if no final answer already exists
             updates_to_state['final_answer'] = "Error: Tool executor was called without pending tool calls from the AI."
        # Even if there's an error, we might have other updates (though unlikely here)
        # We must return the updates_to_state dictionary
        return updates_to_state 

    for tool_call in last_ai_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']

        print(f"  Executing tool: {tool_name} with args: {tool_args} (Call ID: {tool_call_id})")

        if tool_name in tools_map:
            selected_tool = tools_map[tool_name]
            tool_output_str = ""
            try:
                tool_output_content = selected_tool.invoke(tool_args)

                if tool_name == "RetrieverTool":
                    if isinstance(tool_output_content, list) and all(isinstance(doc, Document) for doc in tool_output_content):
                        current_retrieved_documents.extend(tool_output_content)
                        updates_to_state['retrieved_documents'] = current_retrieved_documents
                        tool_output_str = f"Retrieved {len(tool_output_content)} documents. First doc: {tool_output_content[0].page_content[:100]}..." if tool_output_content else "No documents retrieved."
                    else:
                        tool_output_str = f"RetrieverTool returned unexpected data type: {type(tool_output_content)}. Expected List[Document]."
                elif tool_name == "FileReadTool":
                    if isinstance(tool_output_content, str):
                        file_path = tool_args.get('file_path', 'unknown_file') 
                        current_file_contents[file_path] = tool_output_content
                        updates_to_state['file_contents'] = current_file_contents
                        tool_output_str = f"Content of '{file_path}' (length: {len(tool_output_content)}): {tool_output_content[:100]}..." if not tool_output_content.startswith("Error:") else tool_output_content
                    else:
                        tool_output_str = f"FileReadTool returned unexpected data type: {type(tool_output_content)}. Expected str."
                else:
                    tool_output_str = str(tool_output_content)

                tool_messages_for_this_turn.append(ToolMessage(content=tool_output_str, tool_call_id=tool_call_id, name=tool_name))
                print(f"  Tool {tool_name} output (summary for ToolMessage): {tool_output_str[:200]}...")

            except Exception as e:
                print(f"  Error executing tool {tool_name}: {e}")
                error_message = f"Error executing tool {tool_name}: {str(e)}"
                tool_messages_for_this_turn.append(ToolMessage(content=error_message, tool_call_id=tool_call_id, name=tool_name))
        else:
            print(f"  Error: Tool '{tool_name}' not found in tools_map.")
            error_message = f"Error: Tool '{tool_name}' not found."
            tool_messages_for_this_turn.append(ToolMessage(content=error_message, tool_call_id=tool_call_id, name=tool_name))
    
    updates_to_state['intermediate_steps'] = current_intermediate_steps + tool_messages_for_this_turn
    return updates_to_state


def route_after_reasoning(state: AgentStateV2) -> str:
    """
    Conditional routing function for the graph.
    Decides whether to proceed to tool execution or end the graph.
    """
    print("Conditional Edge: route_after_reasoning")

    # Priority 1: If final_answer is already set (even to an empty string), end the graph.
    if state.get("final_answer") is not None:
        print(f"  Decision: final_answer is not None (Value: '{str(state.get('final_answer'))[:50]}...'). Routing to END.")
        return END
    
    # Priority 2: Check the last message in intermediate_steps for tool calls (only if final_answer was None).
    intermediate_steps = state.get('intermediate_steps', [])
    if not intermediate_steps:
        print("  Warning: intermediate_steps is empty and final_answer was None. Routing to END.")
        return END

    last_message_in_intermediate_steps = intermediate_steps[-1]

    if isinstance(last_message_in_intermediate_steps, AIMessage):
        if last_message_in_intermediate_steps.tool_calls:
            print("  Decision: AIMessage has tool_calls. Route to tool_executor_node_v2.")
            return "tool_executor" 
        else:
            # This implies the LLM gave a response without tool calls, but reasoning_node didn't set final_answer.
            # This path should ideally not be hit if reasoning_node correctly sets final_answer for non-tool responses.
            print(f"  Decision: AIMessage has no tool_calls AND final_answer was None. Routing to END.")
            return END
    
    print("  Warning: Last message in intermediate_steps is not AIMessage with tool_calls, or intermediate_steps is empty, AND final_answer was None. Routing to END.")
    return END
