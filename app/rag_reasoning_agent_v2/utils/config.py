from pydantic import BaseModel, Field

class AgentConfigurationV2(BaseModel):
    """
    Configuration settings for the RAG Agent V2.
    """
    llm_model_name: str = Field(default="gemini-2.5-flash-preview-05-20", description="The language model to use for reasoning and generation.") # Updated model
    embedding_model_name: str = Field(default="models/gemini-embedding-exp-03-07", description="The model to use for generating embeddings.") # Updated model
    
    llm_temperature_default: float = Field(default=0.0, description="Default temperature for LLM calls (e.g., reasoning, tool use decisions).")
    # llm_temperature_synthesis might not be strictly needed if one model does all.
    # Keeping it for potential future use or if the reasoning model also does final synthesis with different settings.
    llm_temperature_synthesis: float = Field(default=0.7, description="Temperature for response synthesis LLM calls (if separate).") 
    
    retriever_k: int = Field(default=5, description="Number of documents the main retriever aims to fetch per query/sub-query.") # Adjusted K
    max_sub_queries: int = Field(default=1, description="Maximum number of sub-queries to generate for the retriever. For V2, direct query is often sufficient.") # Adjusted for simpler V2
    faiss_index_path: str = Field(default="data/faiss_index_v2", description="Path to the FAISS vector store index.")
    
if __name__ == '__main__':
    # Example of how to use it
    default_config = AgentConfigurationV2()
    print("Default V2 Configuration:")
    print(f"  LLM Model: {default_config.llm_model_name}")
    print(f"  Embedding Model: {default_config.embedding_model_name}")
    print(f"  Default Temperature: {default_config.llm_temperature_default}")
    print(f"  Synthesis Temperature: {default_config.llm_temperature_synthesis}")
    print(f"  Retriever K: {default_config.retriever_k}")
    print(f"  Max Sub-queries: {default_config.max_sub_queries}")

    custom_config = AgentConfigurationV2(llm_temperature_default=0.2, retriever_k=3)
    print("\nCustom V2 Configuration:")
    print(f"  Default Temperature: {custom_config.llm_temperature_default}")
    print(f"  Retriever K: {custom_config.retriever_k}")
