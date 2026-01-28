"""
Document Processor for Contract PDF Ingestion.
Handles parsing, sectioning, and chunking of contract documents with special handling
for nested sections, subsections, tables (using pdfplumber), and exhibit subsections.
"""

import re
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config


class ContractDocumentProcessor:
    """
    Processes contract PDF documents with intelligent chunking
    that preserves document structure including sections, subsections, and tables.
    """
    
    def __init__(self, pdf_path: Optional[Path] = None):
        """
        Initialize the document processor.
        
        Args:
            pdf_path: Path to the PDF file. Defaults to Config.CONTRACT_PDF_PATH.
        """
        self.pdf_path = pdf_path or Config.CONTRACT_PDF_PATH
        self.raw_text: str = ""
        self.processed_chunks: List[Document] = []
        self.extracted_tables: List[Dict[str, Any]] = []
        
        # Section pattern for main sections (1., 2., 3., etc.)
        self.main_section_pattern = re.compile(r'^(\d+)\.\s+([A-Z][A-Za-z,\s/&-]+)$', re.MULTILINE)
        
        # Subsection pattern (1.1, 2.1, 6.6, etc.)
        self.subsection_pattern = re.compile(r'^(\d+\.\d+)\s+([A-Z][A-Za-z,\s/&()"-]+)', re.MULTILINE)
        
        # Exhibit pattern (Exhibit A, Exhibit B, etc.)
        self.exhibit_pattern = re.compile(r'Exhibit\s+([A-Z])\s*[-—–]?\s*([^\n(]+)', re.MULTILINE)
        
        # Exhibit subsection pattern for Exhibit G and H (G1., G2., H1., H2., etc.)
        self.exhibit_subsection_pattern = re.compile(r'^([GH]\d+[A]?)\.\s+(.+?)(?:\n|$)', re.MULTILINE)
        
    def load_pdf_with_tables(self) -> tuple:
        """
        Load PDF using pdfplumber to extract text and tables.
        
        Returns:
            Tuple of (full_text, list of tables with metadata)
        """
        print(f"Loading PDF from: {self.pdf_path}")
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        all_text = []
        tables_data = []
        
        with pdfplumber.open(str(self.pdf_path)) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text() or ""
                all_text.append(page_text)
                
                # Extract tables from this page
                page_tables = page.extract_tables()
                
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:  # Has header + data
                        # Convert table to structured format
                        table_json = self._table_to_json(table)
                        if table_json:
                            tables_data.append({
                                "table_index": len(tables_data),
                                "table_json": json.dumps(table_json, ensure_ascii=False),
                                "row_count": len(table) - 1,  # Exclude header
                                "col_count": len(table[0]) if table else 0
                            })
        
        self.raw_text = "\n".join(all_text)
        self.extracted_tables = tables_data
        
        print(f"Extracted {len(tables_data)} tables from PDF")
        return self.raw_text, tables_data
    
    def _table_to_json(self, table: List[List]) -> Optional[Dict]:
        """
        Convert a table (list of lists) to a JSON-friendly structure.
        
        Args:
            table: List of rows, where each row is a list of cells
            
        Returns:
            Dictionary with headers and rows, or None if invalid
        """
        if not table or len(table) < 2:
            return None
        
        # Clean cells - replace None with empty string and strip whitespace
        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).strip().replace("\n", " ")
        
        # First row is headers
        headers = [clean_cell(h) for h in table[0]]
        
        # Skip tables with empty headers
        if not any(headers):
            return None
        
        # Convert remaining rows to list of dicts
        rows = []
        for row in table[1:]:
            if row:
                row_dict = {}
                for i, cell in enumerate(row):
                    if i < len(headers) and headers[i]:
                        row_dict[headers[i]] = clean_cell(cell)
                if row_dict:  # Only add non-empty rows
                    rows.append(row_dict)
        
        if not rows:
            return None
        
        return {
            "headers": headers,
            "rows": rows
        }
    
    def _extract_section_info(self, text: str) -> Dict[str, Any]:
        """
        Extract section information from text chunk.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with section metadata
        """
        section_info = {
            "main_section": "",
            "subsection": "",
            "exhibit": "",
            "exhibit_subsection": ""
        }
        
        # Check for main section
        main_match = self.main_section_pattern.search(text)
        if main_match:
            section_info["main_section"] = f"Section {main_match.group(1)}: {main_match.group(2).strip()}"
        
        # Check for subsection
        subsection_matches = list(self.subsection_pattern.finditer(text))
        if subsection_matches:
            first_subsection = subsection_matches[0].group(1)
            section_info["subsection"] = first_subsection
        
        # Check for exhibit
        exhibit_match = self.exhibit_pattern.search(text)
        if exhibit_match:
            exhibit_letter = exhibit_match.group(1)
            exhibit_title = exhibit_match.group(2).strip()
            section_info["exhibit"] = f"Exhibit {exhibit_letter}: {exhibit_title}"
        
        # Check for exhibit subsection (G1, G2, H1, H2, etc.)
        exhibit_subsection_matches = list(self.exhibit_subsection_pattern.finditer(text))
        if exhibit_subsection_matches:
            # Get all exhibit subsections found in this chunk
            subsections = []
            for match in exhibit_subsection_matches:
                subsection_id = match.group(1)
                subsection_title = match.group(2).strip()
                subsections.append(f"{subsection_id}: {subsection_title}")
            section_info["exhibit_subsection"] = "; ".join(subsections[:3])  # Limit to first 3
        
        return section_info
    
    def _find_table_for_chunk(self, chunk_text: str, chunk_index: int) -> Optional[str]:
        """
        Find if there's an extracted table that matches this chunk's content.
        
        Args:
            chunk_text: The chunk text
            chunk_index: Index of the chunk
            
        Returns:
            JSON string of table data if found, None otherwise
        """
        # Check if this chunk likely contains table content
        if not self.extracted_tables:
            return None
        
        # Simple heuristic: assign tables to chunks based on position
        # More sophisticated matching could check for header text in chunk
        table_headers_in_chunk = []
        
        for table_info in self.extracted_tables:
            try:
                table_json = json.loads(table_info["table_json"])
                headers = table_json.get("headers", [])
                # Check if any header appears in the chunk
                for header in headers:
                    if header and len(header) > 3 and header in chunk_text:
                        return table_info["table_json"]
            except:
                continue
        
        return None
    
    def _create_chunk_metadata(self, text: str, chunk_index: int) -> Dict:
        """
        Create rich metadata for a chunk.
        
        Args:
            text: The chunk text
            chunk_index: Index of this chunk
            
        Returns:
            Dictionary of metadata
        """
        section_info = self._extract_section_info(text)
        
        # Find associated table data
        table_json = self._find_table_for_chunk(text, chunk_index)
        
        metadata = {
            "chunk_index": chunk_index,
            "main_section": section_info["main_section"],
            "subsection": section_info["subsection"],
            "exhibit": section_info["exhibit"],
            "exhibit_subsection": section_info["exhibit_subsection"]
        }
        
        # Add table data if found
        if table_json:
            metadata["table_data"] = table_json
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove page markers
        text = re.sub(r'--\s*\d+\s*of\s*\d+\s*--', '', text)
        
        # Normalize whitespace while preserving paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR/parsing issues
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        
        return text.strip()
    
    def process_documents(self) -> List[Document]:
        """
        Process the loaded documents into chunks with rich metadata.
        
        Returns:
            List of processed Document chunks
        """
        # Load PDF with pdfplumber for table extraction
        full_text, tables = self.load_pdf_with_tables()
        full_text = self._clean_text(full_text)
        
        # Create text splitter with separators optimized for contract documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\nExhibit",  # Split on Exhibits
                "\nG1.", "\nG2.", "\nG3.", "\nG4.", "\nG5.",  # Split on G-sections
                "\nG6.", "\nG7.", "\nG8.", "\nG9.", "\nG10.",
                "\nG11.", "\nG12.", "\nG13.",
                "\nH1.", "\nH2.", "\nH3.", "\nH4.", "\nH5.", "\nH6.",  # Split on H-sections
                "\n\n",  # Double newlines
                "\n",    # Single newlines
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters
            ],
            keep_separator=True
        )
        
        # Split the full text
        raw_chunks = text_splitter.split_text(full_text)
        
        print(f"Created {len(raw_chunks)} raw chunks")
        
        # Create Document objects with metadata
        self.processed_chunks = []
        
        for i, chunk_text in enumerate(raw_chunks):
            metadata = self._create_chunk_metadata(chunk_text, i)
            
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            self.processed_chunks.append(doc)
        
        print(f"Processed {len(self.processed_chunks)} chunks with metadata")
        
        # Print exhibit subsection summary
        exhibit_subsections = [
            chunk.metadata.get("exhibit_subsection", "")
            for chunk in self.processed_chunks
            if chunk.metadata.get("exhibit_subsection")
        ]
        if exhibit_subsections:
            print(f"\nFound {len(exhibit_subsections)} chunks with exhibit_subsection metadata")
            unique_subsections = set()
            for es in exhibit_subsections:
                for part in es.split("; "):
                    if part:
                        unique_subsections.add(part.split(":")[0])
            print(f"Unique exhibit subsections: {sorted(unique_subsections)}")
        
        # Print table summary
        table_chunks = [
            chunk for chunk in self.processed_chunks
            if chunk.metadata.get("table_data")
        ]
        print(f"\nChunks with table_data metadata: {len(table_chunks)}")
        
        return self.processed_chunks
    
    def get_chunks(self) -> List[Document]:
        """
        Get processed chunks, processing if necessary.
        
        Returns:
            List of processed Document chunks
        """
        if not self.processed_chunks:
            self.process_documents()
        return self.processed_chunks


def main():
    """Test the document processor."""
    processor = ContractDocumentProcessor()
    chunks = processor.process_documents()
    
    print("\n" + "="*60)
    print("SAMPLE CHUNKS")
    print("="*60)
    
    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {json.dumps({k: v for k, v in chunk.metadata.items() if k != 'table_data'}, indent=2)}")
        if chunk.metadata.get("table_data"):
            print(f"Table data: {chunk.metadata['table_data'][:200]}...")
        print(f"Content preview: {chunk.page_content[:200]}...")
    
    # Show chunks with exhibit_subsection
    print("\n" + "="*60)
    print("CHUNKS WITH EXHIBIT SUBSECTIONS (G or H)")
    print("="*60)
    
    exhibit_sub_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("exhibit_subsection")
    ]
    
    for chunk in exhibit_sub_chunks[:5]:
        print(f"\n--- Exhibit Subsection: {chunk.metadata['exhibit_subsection']} ---")
        print(f"Content preview: {chunk.page_content[:150]}...")
    
    # Show chunks with table data
    print("\n" + "="*60)
    print("CHUNKS WITH TABLE DATA")
    print("="*60)
    
    table_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("table_data")
    ]
    
    for chunk in table_chunks[:3]:
        print(f"\n--- Table Chunk {chunk.metadata['chunk_index']} ---")
        try:
            table_json = json.loads(chunk.metadata["table_data"])
            print(f"Headers: {table_json.get('headers', [])}")
            print(f"Row count: {len(table_json.get('rows', []))}")
        except:
            print(f"Table data: {chunk.metadata['table_data'][:150]}...")


if __name__ == "__main__":
    main()
