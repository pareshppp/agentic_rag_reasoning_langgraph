from pydantic import BaseModel, Field

class AgentConfiguration(BaseModel):
    """
    Configuration settings for the RAG Agent.
    """
    llm_model_name: str = Field(default="gemini-2.5-flash-preview-05-20", description="The language model to use for generation and reasoning.")
    embedding_model_name: str = Field(default="models/gemini-embedding-001", description="The model to use for generating embeddings.")
    
    llm_temperature_default: float = Field(default=0.0, description="Default temperature for LLM calls (e.g., analysis, planning).")
    llm_temperature_synthesis: float = Field(default=0.7, description="Temperature for response synthesis LLM calls.")
    
    retriever_k: int = Field(default=10, description="Number of documents the main retriever aims to fetch per query/sub-query.")
    max_sub_queries: int = Field(default=3, description="Maximum number of sub-queries to generate for the retriever.")

    # Example of how to make it easily usable
    # @classmethod
    # def default(cls):
    #     return cls()

if __name__ == '__main__':
    # Example of how to use it
    default_config = AgentConfiguration()
    print("Default Configuration:")
    print(f"  LLM Model: {default_config.llm_model_name}")
    print(f"  Embedding Model: {default_config.embedding_model_name}")
    print(f"  Default Temperature: {default_config.llm_temperature_default}")
    print(f"  Synthesis Temperature: {default_config.llm_temperature_synthesis}")
    print(f"  Retriever K: {default_config.retriever_k}")
    print(f"  Max Sub-queries: {default_config.max_sub_queries}")

    custom_config = AgentConfiguration(llm_temperature_synthesis=0.9, retriever_k=5)
    print("\nCustom Configuration:")
    print(f"  Synthesis Temperature: {custom_config.llm_temperature_synthesis}")
    print(f"  Retriever K: {custom_config.retriever_k}")
