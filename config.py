"""
Configuration module for the Contract Compliance RAG System.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the RAG system."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone settings
    INDEX_NAME: str = os.getenv("INDEX_NAME", "manulife-assignment")
    
    # OpenAI settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_DIMENSIONS: int = 1536
    
    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    TOP_K_RESULTS: int = 8
    
    # File paths
    BASE_DIR: Path = Path(__file__).parent
    CONTRACT_PDF_PATH: Path = BASE_DIR / "Sample Contract.pdf"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")
        if not cls.INDEX_NAME:
            errors.append("INDEX_NAME is not set")
            
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        return True
    
    @classmethod
    def display(cls) -> None:
        """Display current configuration (masking sensitive values)."""
        print("\n=== Configuration ===")
        print(f"OpenAI API Key: {'*' * 20}...{cls.OPENAI_API_KEY[-4:] if cls.OPENAI_API_KEY else 'NOT SET'}")
        print(f"Pinecone API Key: {'*' * 20}...{cls.PINECONE_API_KEY[-4:] if cls.PINECONE_API_KEY else 'NOT SET'}")
        print(f"Index Name: {cls.INDEX_NAME}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Contract PDF: {cls.CONTRACT_PDF_PATH}")
        print("=====================\n")
