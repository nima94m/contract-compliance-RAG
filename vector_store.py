"""
Vector Store Manager for Pinecone.
Handles creating, indexing, and querying the vector database.
"""

import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import Config


class VectorStoreManager:
    """
    Manages the Pinecone vector store for contract document embeddings.
    """
    
    def __init__(self):
        """Initialize the vector store manager."""
        if not Config.validate():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.vector_store: Optional[PineconeVectorStore] = None
        self.index_name = Config.INDEX_NAME
        
    def index_exists(self) -> bool:
        """Check if the Pinecone index exists."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        return self.index_name in existing_indexes
    
    def create_index(self) -> None:
        """Create the Pinecone index if it doesn't exist."""
        if self.index_exists():
            print(f"Index '{self.index_name}' already exists.")
            return
        
        print(f"Creating index '{self.index_name}'...")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=Config.EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        print(f"Index '{self.index_name}' is ready.")
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        if self.index_exists():
            print(f"Deleting index '{self.index_name}'...")
            self.pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted.")
        else:
            print(f"Index '{self.index_name}' does not exist.")
    
    def clear_index(self) -> None:
        """Clear all vectors from the index."""
        if not self.index_exists():
            print(f"Index '{self.index_name}' does not exist.")
            return
        
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            # Only clear if there are vectors
            if stats.total_vector_count > 0:
                print(f"Clearing {stats.total_vector_count} vectors from index '{self.index_name}'...")
                index.delete(delete_all=True)
                print("Index cleared.")
            else:
                print(f"Index '{self.index_name}' is already empty.")
        except Exception as e:
            print(f"Warning: Could not clear index: {str(e)}")
    
    def ingest_documents(self, documents: List[Document]) -> PineconeVectorStore:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of Document objects to ingest
            
        Returns:
            PineconeVectorStore instance
        """
        # Ensure index exists
        self.create_index()
        
        print(f"Ingesting {len(documents)} documents into Pinecone...")
        
        # Create vector store from documents
        self.vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        print(f"Successfully ingested {len(documents)} documents.")
        
        return self.vector_store
    
    def get_vector_store(self) -> PineconeVectorStore:
        """
        Get the vector store instance, connecting to existing index.
        
        Returns:
            PineconeVectorStore instance
        """
        if self.vector_store is None:
            if not self.index_exists():
                raise ValueError(f"Index '{self.index_name}' does not exist. Please run ingestion first.")
            
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = Config.TOP_K_RESULTS,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant Document objects
        """
        vector_store = self.get_vector_store()
        
        if filter_dict:
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=k
            )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = Config.TOP_K_RESULTS
    ) -> List[tuple]:
        """
        Perform similarity search and return results with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        vector_store = self.get_vector_store()
        results = vector_store.similarity_search_with_score(query=query, k=k)
        return results
    
    def get_index_stats(self) -> dict:
        """Get statistics about the Pinecone index."""
        if not self.index_exists():
            return {"error": f"Index '{self.index_name}' does not exist"}
        
        index = self.pc.Index(self.index_name)
        stats = index.describe_index_stats()
        
        return {
            "index_name": self.index_name,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {}
        }


def main():
    """Test the vector store manager."""
    manager = VectorStoreManager()
    
    # Check index status
    print(f"Index exists: {manager.index_exists()}")
    
    if manager.index_exists():
        stats = manager.get_index_stats()
        print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
