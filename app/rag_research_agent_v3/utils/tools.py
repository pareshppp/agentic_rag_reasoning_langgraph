"""Tools for the RAG Research Agent."""

from typing import Dict, List, Optional, Any, Type
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI


class RetrieverInput(BaseModel):
    """Input for the retriever tool."""
    query: str = Field(description="The query to search for in the vector database")


class FileReadInput(BaseModel):
    """Input for the file read tool."""
    file_path: str = Field(description="The path to the file to read")


class RetrieverTool(BaseTool):
    """Tool for retrieving information from a MongoDB vector database."""
    
    name: str = "retriever_tool"
    description: str = "Use this tool to retrieve relevant information from a MongoDB vector database."
    args_schema: Type[BaseModel] = RetrieverInput
    retriever: BaseRetriever
    
    def _run(self, query: str) -> str:
        """Run the retriever tool."""
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found in the database."
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown source")
            results.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}\n")
        
        return "\n".join(results)


class FileReadTool(BaseTool):
    """Tool for reading file contents."""
    
    name: str = "file_read_tool"
    description: str = "Use this tool to read file contents given a file path."
    args_schema: Type[BaseModel] = FileReadInput
    
    def _run(self, file_path: str) -> str:
        """Run the file read tool."""
        try:
            with open(file_path, "r") as file:
                content = file.read()
            return f"File contents of {file_path}:\n{content}"
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"


def create_retriever(
    mongodb_conn_string: str,
    database_name: str,
    collection_name: str,
    index_name: str,
    embedding_model_name: str = "models/embedding-001",
    llm_model_name: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    use_multi_query: bool = True,
    top_k: int = 5,
) -> BaseRetriever:
    """Create a MongoDB vector retriever."""
    # Create the base MongoDB Atlas vector search retriever
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=mongodb_conn_string,
        namespace=f"{database_name}.{collection_name}",
        index_name=index_name,
        embedding=embedding_model_name,
        text_key="text",
        embedding_key="embedding",
    )
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # If multi-query retriever is requested, wrap the base retriever
    if use_multi_query:
        # Create LLM for query generation
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
        
        # Create multi-query retriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
        return retriever
    
    return base_retriever


def create_tools(
    mongodb_conn_string: str,
    database_name: str,
    collection_name: str,
    index_name: str,
    embedding_model_name: str = "models/embedding-001",
    llm_model_name: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    use_multi_query: bool = True,
    top_k: int = 5,
) -> List[BaseTool]:
    """Create tools for the agent."""
    # Create retriever
    retriever = create_retriever(
        mongodb_conn_string=mongodb_conn_string,
        database_name=database_name,
        collection_name=collection_name,
        index_name=index_name,
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        temperature=temperature,
        use_multi_query=use_multi_query,
        top_k=top_k,
    )
    
    # Create tools
    retriever_tool = RetrieverTool(retriever=retriever)
    file_read_tool = FileReadTool()
    
    return [retriever_tool, file_read_tool]
