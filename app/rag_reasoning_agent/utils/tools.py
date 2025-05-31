import os
import typing
import ast  # For literal_eval

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Attempt to import AgentConfiguration
try:
    from .config import AgentConfiguration
except ImportError:
    # Fallback for cases where script might be run directly for testing (though tests should be separate)
    from rag_reasoning_agent.utils.config import AgentConfiguration

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set - this check can remain at module level
# as it's fundamental for the Google-based tools to operate.
if os.getenv("GOOGLE_API_KEY") is None:
    # This could be a warning print instead of a raise, allowing config to be loaded
    # but tools to fail later if API key is truly needed and not provided.
    # For now, keeping it as a raise to be explicit about dependency.
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# Dummy documents, can be defined at module level as they are static for this example
DUMMY_DOCS = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Gemini is a family of multimodal AI models developed by Google DeepMind."),
    Document(page_content="LangChain provides a framework for developing applications powered by language models."),
    Document(page_content="The FAISS library is written in C++ with Python bindings."),
    Document(page_content="Google's Gemini model can process text, images, and video."),
]

# --- FAISS Retriever Tool Components ---

def _generate_sub_queries_configured(
    query: str, 
    config: AgentConfiguration, 
    llm_instance: ChatGoogleGenerativeAI
) -> typing.List[str]:
    """
    Generates sub-queries based on the input query and configuration, using a provided LLM instance.
    """
    prompt_text = f"""Based on the following user query, generate up to {config.max_sub_queries} sub-queries or alternative search queries that would be effective for retrieving relevant information from a vector database. Return your answer as a Python list of strings.
    Ensure the output is ONLY the Python list of strings, with no other text before or after it. For example: ['query1', 'query2', 'query3']
    Query: {query}"""
    
    response = llm_instance.invoke(prompt_text)
    content = response.content

    try:
        sub_queries_list = ast.literal_eval(content)
        if isinstance(sub_queries_list, list) and all(isinstance(q, str) for q in sub_queries_list):
            return sub_queries_list[:config.max_sub_queries] # Ensure not to exceed max_sub_queries
        else:
            print(f"Warning: Could not parse LLM response for sub-queries as a list of strings. Response: {content}")
            return [query]
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing LLM response for sub-queries: {e}. Response: {content}")
        return [query]

def _retrieve_documents_configured(
    query: str, 
    config: AgentConfiguration, 
    llm_instance: ChatGoogleGenerativeAI, 
    embeddings_instance: GoogleGenerativeAIEmbeddings,
    vector_store_instance: FAISS # Pass the pre-initialized vector store
) -> typing.List[Document]:
    """
    Retrieves relevant documents from the FAISS vector store using configuration.
    Uses a pre-configured LLM for sub-query generation and a pre-configured FAISS store.
    """
    sub_queries = _generate_sub_queries_configured(query, config, llm_instance)
    
    all_retrieved_docs_set = {} 
    
    # The retriever should be re-created with the desired 'k' from config for each call,
    # operating on the provided vector_store_instance.
    retriever = vector_store_instance.as_retriever(search_kwargs={"k": config.retriever_k})

    for sub_query in sub_queries:
        try:
            retrieved_docs = retriever.invoke(sub_query)
            for doc in retrieved_docs:
                if doc.page_content not in all_retrieved_docs_set:
                    all_retrieved_docs_set[doc.page_content] = doc
        except Exception as e:
            print(f"Error retrieving documents for sub-query '{sub_query}': {e}")
            
    return list(all_retrieved_docs_set.values())

def create_faiss_retriever_tool(config: AgentConfiguration) -> Tool:
    """
    Factory function to create a FAISS Retriever Tool configured with AgentConfiguration.
    """
    # Initialize LLM and Embeddings based on config for this tool instance
    llm_for_sub_queries = ChatGoogleGenerativeAI(
        model=config.llm_model_name,
        temperature=config.llm_temperature_default 
    )
    
    embeddings_for_store = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model_name
    )

    # Initialize the FAISS vector store once when the tool is created.
    # This assumes DUMMY_DOCS are relevant for the lifetime of this tool instance.
    # If docs could change, this vector_store would need to be managed differently (e.g., passed in, or re-created more often).
    try:
        vector_store = FAISS.from_documents(DUMMY_DOCS, embeddings_for_store)
        print(f"FAISS vector store initialized successfully for tool with embedding model: {config.embedding_model_name}")
    except Exception as e:
        print(f"Error initializing FAISS vector store: {e}")
        # Fallback or raise error: For now, let it proceed; retrieval will likely fail.
        # A more robust approach might be to not create the tool if this fails.
        vector_store = None 


    # Define the actual function the tool will execute, closing over the configured instances
    def tool_func(query: str) -> typing.List[Document]:
        if not vector_store:
            return [Document(page_content="Error: FAISS vector store not initialized.")]
            
        return _retrieve_documents_configured(
            query=query,
            config=config,
            llm_instance=llm_for_sub_queries,
            embeddings_instance=embeddings_for_store, # Though embeddings instance is used for store creation, not directly here
            vector_store_instance=vector_store
        )

    return Tool(
        name="FaissRetrieverTool",
        func=tool_func,
        description=f"Retrieves relevant documents from a FAISS vector store. Uses {config.embedding_model_name} for embeddings, retrieves {config.retriever_k} docs per query, and generates up to {config.max_sub_queries} sub-queries using {config.llm_model_name}."
    )

# --- File Read Tool Components ---

def read_file_content(file_path: str) -> str:
    """
    Reads the content of a specified file.
    Returns the content as a string.
    Handles FileNotFoundError and returns an error message.
    (This function remains unchanged as it doesn't depend on AgentConfiguration)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at path: {file_path}"
    except Exception as e:
        return f"An unexpected error occurred while reading the file {file_path}: {e}"

def create_file_read_tool(config: AgentConfiguration) -> Tool:
    """
    Factory function to create a File Read Tool.
    Takes AgentConfiguration for consistency but does not use it.
    """
    return Tool(
        name="FileReadTool",
        func=read_file_content,
        description="Reads the content of a specified file. Input should be the full path to the file."
    )

# Example of how these factories would be used (e.g., in agent.py)
if __name__ == '__main__':
    print("Testing tool creation with default configuration...")
    default_config = AgentConfiguration()
    
    # Create tools using factories
    faiss_tool_instance = create_faiss_retriever_tool(default_config)
    file_tool_instance = create_file_read_tool(default_config)

    print(f"\nCreated FAISS Tool: {faiss_tool_instance.name}")
    print(f"  Description: {faiss_tool_instance.description}")
    
    print(f"\nCreated File Read Tool: {file_tool_instance.name}")
    print(f"  Description: {file_tool_instance.description}")

    # Test FAISS tool invocation (requires GOOGLE_API_KEY)
    if os.getenv("GOOGLE_API_KEY"):
        print("\nTesting FAISS tool invocation...")
        test_query = "What is LangGraph?"
        try:
            retrieved_docs = faiss_tool_instance.invoke(test_query)
            print(f"Retrieved {len(retrieved_docs)} documents for query: '{test_query}'")
            for i, doc in enumerate(retrieved_docs):
                print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error invoking FAISS tool: {e}")
    else:
        print("\nSkipping FAISS tool invocation test as GOOGLE_API_KEY is not set.")

    # Test File Read tool invocation
    print("\nTesting File Read tool invocation...")
    dummy_file_for_test = "tools_factory_test.txt"
    try:
        with open(dummy_file_for_test, "w") as f:
            f.write("Hello from tools factory test!")
        
        content = file_tool_instance.invoke(dummy_file_for_test)
        print(f"Content of '{dummy_file_for_test}': {content}")
        assert content == "Hello from tools factory test!"
        
        error_content = file_tool_instance.invoke("non_existent_file.txt")
        print(f"Attempt to read non_existent_file.txt: {error_content}")
        assert "Error: File not found" in error_content
        
    except Exception as e:
        print(f"Error during File Read tool test: {e}")
    finally:
        if os.path.exists(dummy_file_for_test):
            os.remove(dummy_file_for_test)
        print("Cleaned up dummy file.")
    
    print("\nTool factory testing finished.")
