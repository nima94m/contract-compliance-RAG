#!/usr/bin/env python3
"""
Contract Compliance RAG System - CLI Application
================================================

A production-ready RAG system for automated contract compliance checking
using LangChain, OpenAI GPT-4o-mini, and Pinecone.

Usage:
    python main.py ingest          # Ingest contract PDF into vector store
    python main.py query "<question>"  # Query for compliance
    python main.py interactive     # Interactive mode
    python main.py status          # Check system status
    python main.py clear           # Clear the vector store
"""

import argparse
import json
import sys
from typing import Optional

from config import Config
from document_processor import ContractDocumentProcessor
from vector_store import VectorStoreManager
from compliance_checker import ComplianceChecker


class ContractComplianceCLI:
    """Command-line interface for the Contract Compliance RAG System."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.config = Config
        
    def print_banner(self):
        """Print the application banner."""
        banner = """
+===================================================================+
|         Contract Compliance RAG System                            |
|         Powered by LangChain + OpenAI GPT-4o-mini + Pinecone      |
+===================================================================+
        """
        print(banner)
    
    def print_json_result(self, result: dict):
        """Print the compliance result as formatted JSON."""
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    def run_ingest(self):
        """Ingest the contract PDF into the vector store."""
        print("\n[1/3] Loading and processing PDF document...")
        
        try:
            # Process the document
            processor = ContractDocumentProcessor()
            chunks = processor.process_documents()
            
            print(f"\n[2/3] Initializing vector store...")
            vector_store_manager = VectorStoreManager()
            
            # Clear existing data if present
            if vector_store_manager.index_exists():
                print("Clearing existing index data...")
                vector_store_manager.clear_index()
                import time
                time.sleep(2)  # Wait for clear to complete
            
            print(f"\n[3/3] Ingesting {len(chunks)} chunks into Pinecone...")
            vector_store_manager.ingest_documents(chunks)
            
            # Verify ingestion
            import time
            time.sleep(3)  # Wait for indexing
            stats = vector_store_manager.get_index_stats()
            
            print("\n" + "="*60)
            print("INGESTION COMPLETE")
            print("="*60)
            print(f"Document: {Config.CONTRACT_PDF_PATH.name}")
            print(f"Chunks processed: {len(chunks)}")
            print(f"Vectors in index: {stats.get('total_vectors', 'N/A')}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\nError during ingestion: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_query(self, question: str) -> dict:
        """
        Run a compliance query.
        
        Args:
            question: The compliance question to analyze
            
        Returns:
            Compliance result dictionary
        """
        try:
            checker = ComplianceChecker()
            result = checker.check_compliance(question)
            
            # Print formatted result
            print("\n" + "="*60)
            print("COMPLIANCE CHECK RESULT")
            print("="*60)
            print(checker.format_result({"question": question, **result}))
            
            # Print JSON
            self.print_json_result(result)
            
            return result
            
        except Exception as e:
            print(f"\nError during query: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "compliance_state": "non compliant",
                "confidence_percentage": 0,
                "relevant_quotes": [],
                "rationale": f"Error: {str(e)}"
            }
    
    def run_interactive(self):
        """Run interactive query mode."""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Enter compliance questions to analyze.")
        print("Type 'exit', 'quit', or 'q' to exit.")
        print("Type 'help' for example questions.")
        print("="*60)
        
        example_questions = [
            "Does the contract require MFA for privileged access?",
            "What are the data encryption requirements?",
            "What is the incident notification timeline?",
            "Are there requirements for vulnerability scanning?",
            "What are the password management requirements?",
            "Is data required to be stored only in the United States?",
            "What are the backup and disaster recovery requirements?",
            "Are subprocessors required to have security certifications?",
        ]
        
        checker = None
        
        while True:
            try:
                print("\n")
                question = input("Enter question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\nExample compliance questions:")
                    for i, q in enumerate(example_questions, 1):
                        print(f"  {i}. {q}")
                    continue
                
                # Initialize checker on first use
                if checker is None:
                    print("Initializing compliance checker...")
                    checker = ComplianceChecker()
                
                # Run the query
                result = checker.check_compliance(question)
                
                # Print formatted result
                print("\n" + "="*60)
                print("COMPLIANCE CHECK RESULT")
                print("="*60)
                print(checker.format_result({"question": question, **result}))
                
                # Print JSON
                self.print_json_result(result)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    def run_status(self):
        """Check and display system status."""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        
        # Configuration status
        print("\n[Configuration]")
        Config.display()
        
        # Vector store status
        print("\n[Vector Store]")
        try:
            manager = VectorStoreManager()
            
            if manager.index_exists():
                stats = manager.get_index_stats()
                print(f"  Index: {stats['index_name']}")
                print(f"  Status: ACTIVE")
                print(f"  Total vectors: {stats['total_vectors']}")
                print(f"  Dimension: {stats.get('dimension', 'N/A')}")
            else:
                print(f"  Index: {Config.INDEX_NAME}")
                print(f"  Status: NOT CREATED")
                print("  Run 'python main.py ingest' to create and populate the index.")
        except Exception as e:
            print(f"  Error checking vector store: {str(e)}")
        
        # PDF status
        print("\n[Contract Document]")
        if Config.CONTRACT_PDF_PATH.exists():
            size_kb = Config.CONTRACT_PDF_PATH.stat().st_size / 1024
            print(f"  File: {Config.CONTRACT_PDF_PATH.name}")
            print(f"  Size: {size_kb:.1f} KB")
            print(f"  Status: FOUND")
        else:
            print(f"  File: {Config.CONTRACT_PDF_PATH.name}")
            print(f"  Status: NOT FOUND")
        
        print("\n" + "="*60)
    
    def run_clear(self):
        """Clear the vector store."""
        print("\n" + "="*60)
        print("CLEARING VECTOR STORE")
        print("="*60)
        
        try:
            manager = VectorStoreManager()
            
            if not manager.index_exists():
                print(f"Index '{Config.INDEX_NAME}' does not exist. Nothing to clear.")
                return
            
            # Confirm action
            confirm = input(f"Are you sure you want to clear index '{Config.INDEX_NAME}'? (yes/no): ")
            
            if confirm.lower() in ['yes', 'y']:
                manager.clear_index()
                print("Vector store cleared successfully.")
            else:
                print("Operation cancelled.")
                
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="Contract Compliance RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest                    Ingest contract PDF into vector store
  python main.py query "Is MFA required?"  Run a compliance query
  python main.py interactive               Start interactive mode
  python main.py status                    Check system status
  python main.py clear                     Clear the vector store
        """
    )
    
    parser.add_argument(
        'command',
        choices=['ingest', 'query', 'interactive', 'status', 'clear'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='Compliance question (required for query command)'
    )
    
    args = parser.parse_args()
    
    cli = ContractComplianceCLI()
    cli.print_banner()
    
    if args.command == 'ingest':
        success = cli.run_ingest()
        sys.exit(0 if success else 1)
        
    elif args.command == 'query':
        if not args.question:
            print("Error: Please provide a question for the query command.")
            print("Usage: python main.py query \"Your compliance question here\"")
            sys.exit(1)
        cli.run_query(args.question)
        
    elif args.command == 'interactive':
        cli.run_interactive()
        
    elif args.command == 'status':
        cli.run_status()
        
    elif args.command == 'clear':
        cli.run_clear()


if __name__ == "__main__":
    main()
