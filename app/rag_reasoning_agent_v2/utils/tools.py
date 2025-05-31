import os
import typing
import ast  # For literal_eval

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Attempt to import AgentConfigurationV2
try:
    from .config import AgentConfigurationV2
except ImportError:
    # Fallback for testing or direct execution scenarios
    from rag_reasoning_agent_v2.utils.config import AgentConfigurationV2

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# Dummy documents for the retriever tool - these would typically be loaded from a real source
DUMMY_DOCS = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Gemini is a family of multimodal AI models developed by Google DeepMind."),
    Document(page_content="LangChain provides a framework for developing applications powered by language models."),
    Document(page_content="The FAISS library is written in C++ with Python bindings."),
    Document(page_content="Google's Gemini model can process text, images, and video."),
    Document(page_content="The new V2 agent focuses on direct reasoning with access to retriever and file tools."),
    Document(page_content="AgentConfigurationV2 uses gemini-1.5-flash-latest by default."),
]

# --- FAISS Retriever Tool Components ---

def _generate_sub_queries_configured_v2(
    query: str,
    config: AgentConfigurationV2,
    llm_instance: ChatGoogleGenerativeAI
) -> typing.List[str]:
    """
    Generates sub-queries based on the input query and V2 configuration.
    If config.max_sub_queries is 1, it's expected to return the original query or a slight rephrase.
    """
    if config.max_sub_queries <= 0:
        return [query]

    prompt_text = f"""Based on the following user query, generate up to {config.max_sub_queries} search queries (including the original if relevant) that would be effective for retrieving information from a vector database. Return your answer as a Python list of strings.
    Ensure the output is ONLY the Python list of strings, with no other text before or after it. For example: ['query1', 'query2']
    Query: {query}"""

    response = llm_instance.invoke(prompt_text)
    content = response.content

    try:
        sub_queries_list = ast.literal_eval(content)
        if isinstance(sub_queries_list, list) and all(isinstance(q, str) for q in sub_queries_list):
            return sub_queries_list[:config.max_sub_queries]
        else:
            return [query][:config.max_sub_queries]
    except (SyntaxError, ValueError):
        return [query][:config.max_sub_queries]

def _retrieve_documents_configured_v2(
    query: str,
    config: AgentConfigurationV2,
    llm_instance: ChatGoogleGenerativeAI,
    vector_store_instance: FAISS
) -> typing.List[Document]:
    """
    Retrieves relevant documents from the FAISS vector store using V2 configuration.
    """
    sub_queries = _generate_sub_queries_configured_v2(query, config, llm_instance)
    if not sub_queries:
        sub_queries = [query]

    all_retrieved_docs_map = {} 
    retriever = vector_store_instance.as_retriever(search_kwargs={"k": config.retriever_k})

    for sub_q in sub_queries:
        try:
            retrieved_docs = retriever.invoke(sub_q)
            for doc in retrieved_docs:
                if doc.page_content not in all_retrieved_docs_map:
                    all_retrieved_docs_map[doc.page_content] = doc
        except Exception as e:
            print(f"Error retrieving documents for sub-query '{sub_q}': {e}")
    return list(all_retrieved_docs_map.values())

def create_faiss_retriever_tool_v2(config: AgentConfigurationV2, vector_store: FAISS) -> Tool:
    """
    Factory function to create a FAISS Retriever Tool configured with AgentConfigurationV2.
    Accepts a pre-initialized vector_store.
    """
    llm_for_sub_queries = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default
    )

    def tool_func(query: str) -> typing.List[Document]:
        if not vector_store:
            return [Document(page_content="Error: FAISS vector store not provided or initialized.")]
        return _retrieve_documents_configured_v2(
            query=query,
            config=config,
            llm_instance=llm_for_sub_queries,
            vector_store_instance=vector_store
        )

    return Tool(
        name="RetrieverTool",
        func=tool_func,
        description=(
            f"Retrieves relevant documents from a FAISS vector store. "
            f"Uses {config.embedding_model_name} for embeddings (implicitly via the provided vector store). "
            f"Retrieves up to {config.retriever_k} documents. "
            f"May internally generate up to {config.max_sub_queries} query variations using {config.llm_model_name}."
        )
    )

# --- File Read Tool Components ---

def read_file_content(file_path: str) -> str:
    """
    Reads the content of a specified file.
    Returns the content as a string or an error message.
    """
    if ".env" in file_path or ".pyc" in file_path:
         return f"Error: Access to this file path ('{file_path}') is restricted."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at path: {file_path}"
    except IsADirectoryError:
        return f"Error: Expected a file path, but got a directory: {file_path}"
    except Exception as e:
        return f"An unexpected error occurred while reading the file {file_path}: {e}"

def create_file_read_tool_v2(config: AgentConfigurationV2) -> Tool:
    """
    Factory function to create a File Read Tool.
    """
    return Tool(
        name="FileReadTool",
        func=read_file_content,
        description="Reads the content of a specified file. Input should be the full path to the file. Returns file content or an error message."
    )

if __name__ == '__main__':
    print("Testing V2 tool creation with default configuration...")
    v2_config = AgentConfigurationV2()

    if os.getenv("GOOGLE_API_KEY"):
        try:
            embeddings_for_test_store = GoogleGenerativeAIEmbeddings(
                model=v2_config.embedding_model_name
            )
            # Use a temporary directory for FAISS index if saving/loading
            # For in-memory, DUMMY_DOCS is fine.
            test_vector_store = FAISS.from_documents(DUMMY_DOCS, embeddings_for_test_store)
            print(f"FAISS vector store initialized for testing with embedding model: {v2_config.embedding_model_name}")

            faiss_tool_instance_v2 = create_faiss_retriever_tool_v2(v2_config, test_vector_store)
            print(f"\nCreated FAISS Tool (V2): {faiss_tool_instance_v2.name}")
            print(f"  Description: {faiss_tool_instance_v2.description}")

            print("\nTesting FAISS tool (V2) invocation...")
            test_query_v2 = "Information about LangGraph" # Changed query to be more general
            retrieved_docs_v2 = faiss_tool_instance_v2.invoke(test_query_v2)
            print(f"Retrieved {len(retrieved_docs_v2)} documents for query: '{test_query_v2}'")
            for i, doc in enumerate(retrieved_docs_v2):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")

        except Exception as e:
            print(f"Error during FAISS tool V2 testing: {e}")
    else:
        print("\nSkipping FAISS tool (V2) testing as GOOGLE_API_KEY is not set.")

    file_tool_instance_v2 = create_file_read_tool_v2(v2_config)
    print(f"\nCreated File Read Tool (V2): {file_tool_instance_v2.name}")
    print(f"  Description: {file_tool_instance_v2.description}")

    print("\nTesting File Read tool (V2) invocation...")
    dummy_file_path = "test_dummy_file_v2.txt"
    try:
        with open(dummy_file_path, "w") as f:
            f.write("This is content from a test dummy file for V2 agent tools.")
        file_content = file_tool_instance_v2.invoke(dummy_file_path)
        print(f"Content of '{dummy_file_path}': {file_content}")
    finally:
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    non_existent_file_content = file_tool_instance_v2.invoke("non_existent_file_for_v2.txt")
    print(f"Content of 'non_existent_file_for_v2.txt': {non_existent_file_content}")
