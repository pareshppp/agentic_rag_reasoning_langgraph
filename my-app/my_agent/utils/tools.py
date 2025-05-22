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
