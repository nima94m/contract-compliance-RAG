# Contract Compliance RAG System

A production-ready RAG system for automated contract compliance checking. Upload PDF contracts and get structured compliance analysis with AI-powered insights.

## Features

- **Web UI (Streamlit)**: Upload PDFs, run compliance analysis, and chat with the contract
- **Intelligent PDF Processing**: Handles complex contracts with sections, subsections, and tables
- **Table Extraction**: Uses pdfplumber to extract and store table data as structured JSON
- **Structured Output**: Returns compliance assessments in consistent JSON format
- **Section Citations**: Quotes include section/exhibit references (e.g., "Section 6.2", "Exhibit G3")
- **Chat Interface**: Conversational Q&A over contract content
- **Export Results**: Download compliance reports as JSON

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Vector Database | Pinecone |
| Framework | LangChain |
| Embeddings | text-embedding-3-small |
| PDF Processing | pdfplumber |
| Frontend | Streamlit |
| Package Manager | Pipenv |

## Project Structure

```
Manulife assingment/
├── app.py                  # Streamlit web UI (4 tabs)
├── main.py                 # CLI application
├── config.py               # Configuration
├── document_processor.py   # PDF ingestion pipeline
├── vector_store.py         # Pinecone management
├── compliance_checker.py   # RAG compliance engine
├── Pipfile                 # Dependencies
├── .env                    # API keys (not committed)
└── Sample Contract.pdf     # Test document
```

## Architecture Overview

The system uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Ingestion**: PDF is parsed with pdfplumber, split into chunks with section-aware separators, enriched with metadata (section, subsection, exhibit, exhibit_subsection, table_data), and stored in Pinecone with embeddings.

2. **Query**: User question is embedded, similar chunks are retrieved from Pinecone, context is assembled with section headers, and GPT-4o-mini generates a structured compliance assessment.

3. **Output**: JSON response with compliance_state, confidence_percentage, relevant_quotes (with section citations), and rationale.

## Installation

### Prerequisites

- Python 3.10+
- Pipenv (`pip install pipenv`)
- OpenAI API key
- Pinecone API key

### Setup

1. **Create `.env` file**:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   INDEX_NAME=manulife-assignment
   ```

2. **Install dependencies**:
   ```bash
   pipenv install
   ```

3. **Start the application**:
   ```bash
   pipenv run streamlit run app.py
   ```

## Usage

The web UI has 4 tabs:

| Tab | Description |
|-----|-------------|
| **Upload & Analyze** | Upload PDF or use sample, run 5 compliance questions |
| **Custom Query** | Ask any compliance question with structured output |
| **Chat** | Conversational Q&A about the contract |
| **Results Export** | Download JSON report |

## Output Format

```json
{
  "compliance_state": "fully compliant",
  "confidence_percentage": 95,
  "relevant_quotes": [
    "Section 6.2: Vendor will enforce MFA for privileged accounts...",
    "Exhibit G3 (IAM-01): MFA required for all privileged access..."
  ],
  "rationale": "The contract explicitly requires MFA..."
}
```

### Compliance States

| State | Description |
|-------|-------------|
| `fully compliant` | Contract explicitly meets all aspects of the requirement |
| `partially compliant` | Contract addresses some but not all aspects |
| `non compliant` | Contract does not address the requirement |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `PINECONE_API_KEY` | Required | Pinecone API key |
| `INDEX_NAME` | `manulife-assignment` | Pinecone index name |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `8` | Chunks to retrieve |

## Dependencies

- `langchain` - LLM orchestration framework
- `langchain-openai` - OpenAI integration
- `langchain-pinecone` - Pinecone integration
- `langchain-community` - Community integrations
- `langchain-text-splitters` - Text chunking
- `openai` - OpenAI API client
- `pinecone-client` - Pinecone API client
- `pdfplumber` - PDF and table extraction
- `pypdf` - PDF parsing
- `streamlit` - Web UI framework
- `python-dotenv` - Environment variables
- `tiktoken` - Token counting
- `pydantic` - Data validation

## License

This project is for educational and assessment purposes.
