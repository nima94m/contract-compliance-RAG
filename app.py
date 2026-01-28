"""
Contract Compliance RAG System - Streamlit UI
==============================================

A web interface for automated contract compliance checking.
Upload a PDF contract and get structured compliance analysis.
Includes chat functionality for conversational Q&A.
"""

import streamlit as st
import json
import time
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from document_processor import ContractDocumentProcessor
from vector_store import VectorStoreManager
from compliance_checker import ComplianceChecker


# Page configuration
st.set_page_config(
    page_title="Contract Compliance Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .compliant {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .partial {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .non-compliant {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .quote-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E3A5F;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-style: italic;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Define the 5 compliance questions from the assignment
COMPLIANCE_QUESTIONS = [
    {
        "id": 1,
        "title": "Password Management",
        "question": """Password Management. The contract must require a documented password standard covering password length/strength, prohibition of default and known-compromised passwords, secure storage (no plaintext; salted hashing if stored), brute-force protections (lockout/rate limiting), prohibition on password sharing, vaulting of privileged credentials/recovery codes, and time-based rotation for break-glass credentials. Based on the contract language and exhibits, what is the compliance state for Password Management?"""
    },
    {
        "id": 2,
        "title": "IT Asset Management",
        "question": """IT Asset Management. The contract must require an in-scope asset inventory (including cloud accounts/subscriptions, workloads, databases, security tooling), define minimum inventory fields, require at least quarterly reconciliation/review, and require secure configuration baselines with drift remediation and prohibition of insecure defaults. Based on the contract language and exhibits, what is the compliance state for IT Asset Management?"""
    },
    {
        "id": 3,
        "title": "Security Training & Background Checks",
        "question": """Security Training & Background Checks. The contract must require security awareness training on hire and at least annually, and background screening for personnel with access to Company Data to the extent permitted by law, including maintaining a screening policy and attestation/evidence. Based on the contract language and exhibits, what is the compliance state for Security Training and Background Checks?"""
    },
    {
        "id": 4,
        "title": "Data in Transit Encryption",
        "question": """Data in Transit Encryption. The contract must require encryption of Company Data in transit using TLS 1.2+ (preferably TLS 1.3 where feasible) for Company-to-Service traffic, administrative access pathways, and applicable Service-to-Subprocessor transfers, with certificate management and avoidance of insecure cipher suites. Based on the contract language and exhibits, what is the compliance state for Data in Transit Encryption?"""
    },
    {
        "id": 5,
        "title": "Network Authentication & Authorization Protocols",
        "question": """Network Authentication & Authorization Protocols. The contract must specify the authentication mechanisms (e.g., SAML SSO for users, OAuth/token-based for APIs), require MFA for privileged/production access, require secure admin pathways (bastion/secure gateway) with session logging, and require RBAC authorization. Based on the contract language and exhibits, what is the compliance state for Network Authentication and Authorization Protocols?"""
    }
]


def get_compliance_badge(state: str) -> str:
    """Return HTML badge for compliance state."""
    state_lower = state.lower()
    if "fully" in state_lower:
        return '<span style="background-color: #28a745; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold;">FULLY COMPLIANT</span>'
    elif "partial" in state_lower:
        return '<span style="background-color: #ffc107; color: black; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold;">PARTIALLY COMPLIANT</span>'
    else:
        return '<span style="background-color: #dc3545; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold;">NON-COMPLIANT</span>'


def get_confidence_color(confidence: int) -> str:
    """Return color based on confidence level."""
    if confidence >= 90:
        return "#28a745"
    elif confidence >= 70:
        return "#17a2b8"
    elif confidence >= 50:
        return "#ffc107"
    else:
        return "#dc3545"


def process_uploaded_pdf(uploaded_file) -> bool:
    """Process an uploaded PDF file and ingest into vector store."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)
        
        # Process the document
        processor = ContractDocumentProcessor(pdf_path=tmp_path)
        chunks = processor.process_documents()
        
        # Initialize vector store
        vector_store_manager = VectorStoreManager()
        
        # Clear existing data
        if vector_store_manager.index_exists():
            vector_store_manager.clear_index()
            time.sleep(2)
        
        # Ingest documents
        vector_store_manager.ingest_documents(chunks)
        time.sleep(3)  # Wait for indexing
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return True, len(chunks)
        
    except Exception as e:
        return False, str(e)


def run_compliance_analysis(questions: List[Dict]) -> List[Dict[str, Any]]:
    """Run compliance analysis for all questions."""
    checker = ComplianceChecker()
    results = []
    
    for q in questions:
        result = checker.check_compliance(q["question"])
        result["id"] = q["id"]
        result["title"] = q["title"]
        result["question"] = q["question"]
        results.append(result)
    
    return results


def display_compliance_result(result: Dict[str, Any]):
    """Display a single compliance result in a formatted card."""
    with st.container():
        # Header with title and badge
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {result['id']}. {result['title']}")
        with col2:
            st.markdown(get_compliance_badge(result['compliance_state']), unsafe_allow_html=True)
        
        # Confidence meter
        confidence = result['confidence_percentage']
        st.markdown(f"**Confidence:** {confidence}%")
        st.progress(confidence / 100)
        
        # Relevant quotes
        st.markdown("**Relevant Quotes:**")
        for quote in result.get('relevant_quotes', []):
            st.markdown(f'<div class="quote-box">"{quote}"</div>', unsafe_allow_html=True)
        
        # Rationale
        st.markdown("**Rationale:**")
        st.info(result.get('rationale', 'No rationale provided.'))
        
        # JSON expander
        with st.expander("View JSON Output"):
            json_output = {
                "compliance_state": result['compliance_state'],
                "confidence_percentage": result['confidence_percentage'],
                "relevant_quotes": result.get('relevant_quotes', []),
                "rationale": result.get('rationale', '')
            }
            st.json(json_output)
        
        st.divider()


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">Contract Compliance Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a contract PDF and analyze compliance with security requirements using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool analyzes contract documents for compliance with security requirements using:
        - **LangChain** for orchestration
        - **OpenAI GPT-4o-mini** for analysis
        - **Pinecone** for vector storage
        """)
        
        st.divider()
        
        st.header("Compliance Questions")
        st.markdown("The system analyzes 5 key compliance areas:")
        for q in COMPLIANCE_QUESTIONS:
            st.markdown(f"**{q['id']}.** {q['title']}")
        
        st.divider()
        
        # System status
        st.header("System Status")
        try:
            manager = VectorStoreManager()
            if manager.index_exists():
                stats = manager.get_index_stats()
                st.success(f"Vector Store: Active ({stats['total_vectors']} vectors)")
            else:
                st.warning("Vector Store: Not initialized")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Analyze", "Custom Query", "Chat", "Results Export"])
    
    with tab1:
        st.header("Upload Contract PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a contract PDF document for compliance analysis"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_sample = st.checkbox("Use sample contract (Sample Contract.pdf)", value=False)
        
        with col2:
            run_analysis = st.button("Run Compliance Analysis", type="primary", use_container_width=True)
        
        if run_analysis:
            if not uploaded_file and not use_sample:
                st.error("Please upload a PDF file or select the sample contract.")
            else:
                # Processing status container
                status_container = st.container()
                
                with status_container:
                    # Step 1: Process PDF
                    with st.status("Processing contract...", expanded=True) as status:
                        st.write("Loading PDF document...")
                        
                        if use_sample:
                            # Use the existing sample contract
                            st.write("Using sample contract: Sample Contract.pdf")
                            processor = ContractDocumentProcessor()
                            chunks = processor.process_documents()
                            
                            st.write("Initializing vector store...")
                            vector_store_manager = VectorStoreManager()
                            
                            if vector_store_manager.index_exists():
                                stats = vector_store_manager.get_index_stats()
                                if stats['total_vectors'] > 0:
                                    st.write(f"Using existing index with {stats['total_vectors']} vectors")
                                else:
                                    st.write("Ingesting document into vector store...")
                                    vector_store_manager.ingest_documents(chunks)
                                    time.sleep(3)
                            else:
                                st.write("Creating index and ingesting document...")
                                vector_store_manager.ingest_documents(chunks)
                                time.sleep(3)
                            
                            chunk_count = len(chunks)
                        else:
                            # Process uploaded file
                            st.write(f"Processing: {uploaded_file.name}")
                            success, result = process_uploaded_pdf(uploaded_file)
                            
                            if not success:
                                st.error(f"Error processing PDF: {result}")
                                st.stop()
                            
                            chunk_count = result
                        
                        st.write(f"Document processed: {chunk_count} chunks created")
                        status.update(label="Document processed!", state="complete")
                    
                    # Step 2: Run compliance analysis
                    st.divider()
                    st.header("Compliance Analysis Results")
                    
                    progress_bar = st.progress(0)
                    results = []
                    
                    checker = ComplianceChecker()
                    
                    for i, q in enumerate(COMPLIANCE_QUESTIONS):
                        with st.spinner(f"Analyzing: {q['title']}..."):
                            result = checker.check_compliance(q["question"])
                            result["id"] = q["id"]
                            result["title"] = q["title"]
                            result["question"] = q["question"]
                            results.append(result)
                        
                        progress_bar.progress((i + 1) / len(COMPLIANCE_QUESTIONS))
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    
                    # Display summary metrics
                    st.subheader("Summary")
                    
                    fully_compliant = sum(1 for r in results if "fully" in r['compliance_state'].lower())
                    partial = sum(1 for r in results if "partial" in r['compliance_state'].lower())
                    non_compliant = len(results) - fully_compliant - partial
                    avg_confidence = sum(r['confidence_percentage'] for r in results) / len(results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Fully Compliant", fully_compliant, delta=None)
                    with col2:
                        st.metric("Partially Compliant", partial, delta=None)
                    with col3:
                        st.metric("Non-Compliant", non_compliant, delta=None)
                    with col4:
                        st.metric("Avg. Confidence", f"{avg_confidence:.0f}%", delta=None)
                    
                    st.divider()
                    
                    # Display individual results
                    for result in results:
                        display_compliance_result(result)
    
    with tab2:
        st.header("Custom Compliance Query")
        st.markdown("Ask any compliance-related question about the uploaded contract.")
        
        custom_question = st.text_area(
            "Enter your compliance question:",
            height=100,
            placeholder="e.g., Does the contract require multi-factor authentication for all privileged accounts?"
        )
        
        if st.button("Analyze", type="primary"):
            if not custom_question:
                st.error("Please enter a question.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        checker = ComplianceChecker()
                        result = checker.check_compliance(custom_question)
                        
                        # Display result
                        st.markdown("### Result")
                        st.markdown(get_compliance_badge(result['compliance_state']), unsafe_allow_html=True)
                        
                        st.markdown(f"**Confidence:** {result['confidence_percentage']}%")
                        st.progress(result['confidence_percentage'] / 100)
                        
                        st.markdown("**Relevant Quotes:**")
                        for quote in result.get('relevant_quotes', []):
                            st.markdown(f'<div class="quote-box">"{quote}"</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Rationale:**")
                        st.info(result.get('rationale', 'No rationale provided.'))
                        
                        st.markdown("**JSON Output:**")
                        st.json(result)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("Chat with Contract")
        st.markdown("Have a conversation about the contract document. Ask any question!")
        
        # Initialize chat history in session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Check if vector store is ready
        try:
            manager = VectorStoreManager()
            if not manager.index_exists() or manager.get_index_stats()['total_vectors'] == 0:
                st.warning("Please upload and process a contract document first in the 'Upload & Analyze' tab.")
            else:
                # Display chat messages
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask a question about the contract..."):
                    # Add user message to chat history
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                # Get relevant context from vector store
                                docs = manager.similarity_search(prompt, k=Config.TOP_K_RESULTS)
                                
                                # Format context
                                context_parts = []
                                for i, doc in enumerate(docs, 1):
                                    metadata = doc.metadata
                                    section_info = []
                                    if metadata.get("main_section"):
                                        section_info.append(metadata["main_section"])
                                    if metadata.get("subsection"):
                                        section_info.append(f"Subsection {metadata['subsection']}")
                                    if metadata.get("exhibit"):
                                        section_info.append(metadata["exhibit"])
                                    if metadata.get("exhibit_subsection"):
                                        section_info.append(f"Exhibit Subsection: {metadata['exhibit_subsection']}")
                                    
                                    header = f"[Source {i}] "
                                    if section_info:
                                        header += " | ".join(section_info)
                                    context_parts.append(f"{header}\n{doc.page_content}\n")
                                
                                context = "\n---\n".join(context_parts)
                                
                                # Create chat prompt
                                chat_prompt = ChatPromptTemplate.from_messages([
                                    ("system", """You are a helpful assistant that answers questions about contract documents. 
You have access to relevant sections of the contract and should provide accurate, helpful answers based on the contract content.

Guidelines:
1. Base your answers on the provided contract context
2. If the answer is in the contract, cite the relevant section or exhibit
3. If the answer is not in the provided context, say so honestly
4. Be conversational and helpful
5. For compliance-related questions, indicate whether the contract addresses the topic
6. Keep answers clear and concise unless more detail is requested

CONTRACT CONTEXT:
{context}"""),
                                    ("human", "{question}")
                                ])
                                
                                # Initialize LLM
                                llm = ChatOpenAI(
                                    model=Config.LLM_MODEL,
                                    temperature=0.3,
                                    openai_api_key=Config.OPENAI_API_KEY
                                )
                                
                                # Generate response
                                messages = chat_prompt.format_messages(
                                    context=context,
                                    question=prompt
                                )
                                response = llm.invoke(messages)
                                assistant_response = response.content
                                
                                # Display response
                                st.markdown(assistant_response)
                                
                                # Add to chat history
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": assistant_response
                                })
                                
                            except Exception as e:
                                error_msg = f"Error generating response: {str(e)}"
                                st.error(error_msg)
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                
                # Clear chat button
                if st.session_state.chat_messages:
                    if st.button("Clear Chat History"):
                        st.session_state.chat_messages = []
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Error initializing chat: {str(e)}")
    
    with tab4:
        st.header("Export Results")
        
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            
            # Format for export
            export_data = {
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": len(results),
                "summary": {
                    "fully_compliant": sum(1 for r in results if "fully" in r['compliance_state'].lower()),
                    "partially_compliant": sum(1 for r in results if "partial" in r['compliance_state'].lower()),
                    "non_compliant": sum(1 for r in results if "non" in r['compliance_state'].lower()),
                    "average_confidence": sum(r['confidence_percentage'] for r in results) / len(results)
                },
                "results": [
                    {
                        "id": r['id'],
                        "compliance_question": r['title'],
                        "compliance_state": r['compliance_state'],
                        "confidence_percentage": r['confidence_percentage'],
                        "relevant_quotes": r.get('relevant_quotes', []),
                        "rationale": r.get('rationale', '')
                    }
                    for r in results
                ]
            }
            
            # Display preview
            st.subheader("Results Preview")
            st.json(export_data)
            
            # Download button
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_str,
                file_name="compliance_analysis_report.json",
                mime="application/json"
            )
            
            # Also show as table
            st.subheader("Results Table")
            table_data = []
            for r in results:
                table_data.append({
                    "#": r['id'],
                    "Compliance Question": r['title'],
                    "State": r['compliance_state'].upper(),
                    "Confidence": f"{r['confidence_percentage']}%"
                })
            st.table(table_data)
            
        else:
            st.info("No results available. Please run an analysis first in the 'Upload & Analyze' tab.")


if __name__ == "__main__":
    main()
