import os
import typing
import ast  # For literal_eval

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# 1. Initialize Embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# 2. Dummy Document Setup
docs = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="Gemini is a family of multimodal AI models developed by Google DeepMind."),
    Document(page_content="LangChain provides a framework for developing applications powered by language models."),
    Document(page_content="The FAISS library is written in C++ with Python bindings."),
    Document(page_content="Google's Gemini model can process text, images, and video."),
]

# 3. FAISS Vector Store Initialization
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# 4. Sub-query Generation Function
def generate_sub_queries(query: str) -> typing.List[str]:
    """
    Generates sub-queries or alternative search queries based on the input query.
    """
    prompt = f"""Based on the following user query, generate three sub-queries or alternative search queries that would be effective for retrieving relevant information from a vector database. Return your answer as a Python list of strings.
    Ensure the output is ONLY the Python list of strings, with no other text before or after it. For example: ['query1', 'query2', 'query3']
    Query: {query}"""
    
    response = llm.invoke(prompt)
    content = response.content

    try:
        # The response content might be a string representation of a list.
        # We use ast.literal_eval to safely evaluate it.
        sub_queries_list = ast.literal_eval(content)
        if isinstance(sub_queries_list, list) and all(isinstance(q, str) for q in sub_queries_list):
            return sub_queries_list
        else:
            # Fallback if parsing fails or is not a list of strings
            print(f"Warning: Could not parse LLM response for sub-queries as a list of strings. Response: {content}")
            return [query] # Fallback to the original query
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing LLM response for sub-queries: {e}. Response: {content}")
        # Fallback if parsing fails
        return [query] # Fallback to the original query

# 5. Retriever Function (Core Logic for the Tool)
def retrieve_documents(query: str) -> typing.List[Document]:
    """
    Retrieves relevant documents from the FAISS vector store.
    It breaks the query into sub-queries, retrieves documents for each,
    and returns a unique list of documents.
    """
    sub_queries = generate_sub_queries(query)
    
    all_retrieved_docs_set = {} # Using dict to store docs by page_content for uniqueness

    for sub_query in sub_queries:
        try:
            # Using retriever.invoke() which is the current standard
            retrieved_docs = retriever.invoke(sub_query)
            for doc in retrieved_docs:
                if doc.page_content not in all_retrieved_docs_set:
                    all_retrieved_docs_set[doc.page_content] = doc
        except Exception as e:
            print(f"Error retrieving documents for sub-query '{sub_query}': {e}")
            # Optionally, re-try with get_relevant_documents if invoke fails, though invoke is preferred
            # try:
            #     retrieved_docs = retriever.get_relevant_documents(sub_query)
            #     for doc in retrieved_docs:
            #         if doc.page_content not in all_retrieved_docs_set:
            #             all_retrieved_docs_set[doc.page_content] = doc
            # except Exception as e2:
            #     print(f"Error also with get_relevant_documents for sub-query '{sub_query}': {e2}")


    return list(all_retrieved_docs_set.values())

# 6. Create Langchain Tool
faiss_retriever_tool = Tool(
    name="FaissRetrieverTool",
    func=retrieve_documents,
    description="Retrieves relevant documents from the FAISS vector store based on the user query. It breaks the query into sub-queries for better results."
)

if __name__ == '__main__':
    # Example usage (requires GOOGLE_API_KEY to be set in .env)
    # Create a .env file in the my-app directory with:
    # GOOGLE_API_KEY="your_actual_api_key"

    print("Testing FAISS Retriever Tool...")

    # Test sub-query generation
    sample_query_for_subqueries = "What are the features of LangGraph and FAISS?"
    print(f"\nGenerating sub-queries for: '{sample_query_for_subqueries}'")
    generated_subqueries = generate_sub_queries(sample_query_for_subqueries)
    print(f"Generated sub-queries: {generated_subqueries}")

    # Test document retrieval
    sample_query_for_retrieval = "Tell me about LangGraph and Gemini."
    print(f"\nRetrieving documents for query: '{sample_query_for_retrieval}'")
    retrieved_docs = retrieve_documents(sample_query_for_retrieval)
    
    if retrieved_docs:
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1}:")
            print(f"  Page Content: {doc.page_content}")
            if doc.metadata:
                print(f"  Metadata: {doc.metadata}")
    else:
        print("No documents retrieved.")

    # Test with a query that should ideally use sub-queries
    complex_query = "Compare LangGraph's state management with traditional methods and explain FAISS indexing."
    print(f"\nRetrieving documents for complex query: '{complex_query}'")
    retrieved_docs_complex = retrieve_documents(complex_query)

    if retrieved_docs_complex:
        print("\nRetrieved Documents (Complex Query):")
        for i, doc in enumerate(retrieved_docs_complex):
            print(f"Document {i+1}:")
            print(f"  Page Content: {doc.page_content}")
    else:
        print("No documents retrieved for complex query.")

    print("\nFAISS Retriever Tool setup complete.")
    print("Tool Name:", faiss_retriever_tool.name)
    print("Tool Description:", faiss_retriever_tool.description)

    # Example of invoking the tool directly (as Langchain would)
    # Note: This assumes GOOGLE_API_KEY is available.
    # If running this file directly, ensure .env is in the same directory or GOOGLE_API_KEY is in the environment.
    # print("\nTesting tool invocation:")
    # tool_result = faiss_retriever_tool.invoke("What is Gemini?")
    # print("Tool Result:", tool_result)


# 7. File Read Function
def read_file_content(file_path: str) -> str:
    """
    Reads the content of a specified file.
    Returns the content as a string.
    Handles FileNotFoundError and returns an error message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at path: {file_path}"
    except Exception as e:
        return f"An unexpected error occurred while reading the file {file_path}: {e}"

# 8. Create Langchain Tool for File Reading
file_read_tool = Tool(
    name="FileReadTool",
    func=read_file_content,
    description="Reads the content of a specified file. Input should be the full path to the file."
)

if __name__ == '__main__':
    # ... (previous test code remains the same)

    print("\nTesting File Read Tool...")
    # Create a dummy file for testing
    dummy_file_path = "dummy_test_file.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test file for the FileReadTool.\nHello World!")
    
    # Test reading an existing file
    print(f"Reading content from: {dummy_file_path}")
    content = file_read_tool.invoke(dummy_file_path)
    print("File Content:")
    print(content)

    # Test reading a non-existing file
    non_existent_file_path = "non_existent_file.txt"
    print(f"\nAttempting to read content from non-existent file: {non_existent_file_path}")
    error_message = file_read_tool.invoke(non_existent_file_path)
    print("Tool Output (Error):")
    print(error_message)

    # Clean up the dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
    print(f"\nCleaned up {dummy_file_path}")
