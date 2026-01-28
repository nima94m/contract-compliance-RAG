"""
Compliance Checker Module.
Uses RAG to analyze contract compliance and returns structured JSON responses.
"""

import json
from typing import Dict, List, Any, Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import Config
from vector_store import VectorStoreManager


class ComplianceState(str, Enum):
    """Enumeration of possible compliance states."""
    FULLY_COMPLIANT = "fully compliant"
    PARTIALLY_COMPLIANT = "partially compliant"
    NON_COMPLIANT = "non compliant"


class ComplianceResult(BaseModel):
    """Structured compliance check result."""
    compliance_state: str = Field(
        description="One of: 'fully compliant', 'partially compliant', or 'non compliant'"
    )
    confidence_percentage: int = Field(
        ge=0, le=100,
        description="Confidence level from 0-100%"
    )
    relevant_quotes: List[str] = Field(
        description="Direct quotes from the contract supporting the assessment"
    )
    rationale: str = Field(
        description="Detailed explanation of the compliance determination"
    )


class ComplianceChecker:
    """
    RAG-based compliance checker that analyzes contract documents
    and returns structured compliance assessments.
    """
    
    def __init__(self):
        """Initialize the compliance checker."""
        if not Config.validate():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,  # Low temperature for consistent analysis
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager()
        
        # Compliance analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert contract compliance analyst specializing in information security and technology risk assessments. Your task is to analyze contract clauses and determine compliance status.

IMPORTANT INSTRUCTIONS:
1. Base your analysis ONLY on the provided contract context
2. Be precise and cite specific sections/clauses when possible
3. Consider both explicit requirements and implied obligations
4. If the contract doesn't address a question directly, indicate uncertainty
5. Always provide direct quotes to support your findings

COMPLIANCE STATES:
- "fully compliant": The contract explicitly addresses and meets all aspects of the requirement
- "partially compliant": The contract addresses some but not all aspects, or meets requirements with conditions/exceptions
- "non compliant": The contract does not address the requirement, or explicitly contradicts it

CONFIDENCE GUIDELINES:
- 90-100%: Clear, explicit language directly addresses the question
- 70-89%: Strong implications or related clauses support the conclusion
- 50-69%: Indirect evidence or partial coverage
- Below 50%: Limited or ambiguous evidence; indicate uncertainty

RELEVANT QUOTES FORMAT:
Each quote MUST include the section or exhibit reference at the beginning. Format examples:
- "Section 6.2: Vendor will enforce multi-factor authentication..."
- "Section 8.2 (Remote Administration): Vendor will restrict administrative access..."
- "Exhibit G, G3 (IAM-01): Enforce MFA for privileged access..."
- "Exhibit G13 (NET-01): Secure admin network access..."
- "Subsection 7.2: Vendor will enforce encryption in transit..."

Always prefix each quote with its source location (Section X.X, Exhibit X, or Exhibit Subsection like G1, G13, H1, etc.).

OUTPUT FORMAT:
You must respond with a valid JSON object with exactly these fields:
{{
    "compliance_state": "<one of: fully compliant, partially compliant, non compliant>",
    "confidence_percentage": <integer 0-100>,
    "relevant_quotes": ["<Section/Exhibit reference>: <quote text>", ...],
    "rationale": "<detailed explanation>"
}}"""),
            ("human", """Based on the following contract sections, analyze this compliance question:

QUESTION: {question}

CONTRACT CONTEXT:
{context}

Provide your compliance assessment as a JSON object. Remember to include the section/exhibit reference at the beginning of each relevant quote.""")
        ])
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
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
            else:
                header += f"Chunk {metadata.get('chunk_index', i)}"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured format.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed compliance result dictionary
        """
        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON block in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Return error response
                    result = {
                        "compliance_state": "non compliant",
                        "confidence_percentage": 0,
                        "relevant_quotes": [],
                        "rationale": f"Error parsing response: {response_text[:500]}"
                    }
            else:
                result = {
                    "compliance_state": "non compliant",
                    "confidence_percentage": 0,
                    "relevant_quotes": [],
                    "rationale": f"Error parsing response: {response_text[:500]}"
                }
        
        # Validate and normalize compliance state
        state = result.get("compliance_state", "").lower().strip()
        valid_states = ["fully compliant", "partially compliant", "non compliant"]
        
        if state not in valid_states:
            # Try to match partial strings
            if "fully" in state or "complete" in state:
                state = "fully compliant"
            elif "partial" in state:
                state = "partially compliant"
            else:
                state = "non compliant"
        
        result["compliance_state"] = state
        
        # Ensure confidence is an integer in valid range
        confidence = result.get("confidence_percentage", 50)
        if isinstance(confidence, str):
            confidence = int(confidence.replace("%", ""))
        result["confidence_percentage"] = max(0, min(100, int(confidence)))
        
        # Ensure relevant_quotes is a list
        quotes = result.get("relevant_quotes", [])
        if isinstance(quotes, str):
            quotes = [quotes]
        result["relevant_quotes"] = quotes
        
        # Ensure rationale exists
        if not result.get("rationale"):
            result["rationale"] = "No detailed rationale provided."
        
        return result
    
    def check_compliance(
        self,
        question: str,
        k: int = Config.TOP_K_RESULTS
    ) -> Dict[str, Any]:
        """
        Check compliance based on a question.
        
        Args:
            question: The compliance question to analyze
            k: Number of context documents to retrieve
            
        Returns:
            Structured compliance result dictionary
        """
        print(f"\n{'='*60}")
        print(f"Analyzing compliance question...")
        print(f"{'='*60}")
        
        # Retrieve relevant documents
        print("Retrieving relevant contract sections...")
        documents = self.vector_store_manager.similarity_search(question, k=k)
        
        if not documents:
            return {
                "compliance_state": "non compliant",
                "confidence_percentage": 0,
                "relevant_quotes": [],
                "rationale": "No relevant contract sections found for this question."
            }
        
        print(f"Found {len(documents)} relevant sections")
        
        # Format context
        context = self._format_context(documents)
        
        # Generate analysis
        print("Generating compliance analysis...")
        messages = self.analysis_prompt.format_messages(
            question=question,
            context=context
        )
        
        response = self.llm.invoke(messages)
        
        # Parse and return result
        result = self._parse_llm_response(response.content)
        
        return result
    
    def check_compliance_batch(
        self,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check compliance for multiple questions.
        
        Args:
            questions: List of compliance questions
            
        Returns:
            List of compliance results
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question...")
            result = self.check_compliance(question)
            result["question"] = question
            results.append(result)
        
        return results
    
    def format_result(self, result: Dict[str, Any], include_question: bool = True) -> str:
        """
        Format a compliance result for display.
        
        Args:
            result: Compliance result dictionary
            include_question: Whether to include the question in output
            
        Returns:
            Formatted string representation
        """
        output = []
        
        if include_question and "question" in result:
            output.append(f"Question: {result['question']}")
            output.append("-" * 50)
        
        # Compliance state with visual indicator (ASCII-safe)
        state = result["compliance_state"]
        state_icon = {
            "fully compliant": "[PASS]",
            "partially compliant": "[PARTIAL]",
            "non compliant": "[FAIL]"
        }.get(state, "[?]")
        
        output.append(f"Compliance State: {state_icon} {state.upper()}")
        output.append(f"Confidence: {result['confidence_percentage']}%")
        
        output.append("\nRelevant Quotes:")
        for i, quote in enumerate(result.get("relevant_quotes", []), 1):
            # Truncate long quotes
            if len(quote) > 300:
                quote = quote[:300] + "..."
            output.append(f"  {i}. \"{quote}\"")
        
        output.append(f"\nRationale:\n{result['rationale']}")
        
        return "\n".join(output)
    
    def get_json_result(self, result: Dict[str, Any]) -> str:
        """
        Get the result as a formatted JSON string.
        
        Args:
            result: Compliance result dictionary
            
        Returns:
            Pretty-printed JSON string
        """
        return json.dumps(result, indent=2, ensure_ascii=False)


def main():
    """Test the compliance checker."""
    checker = ComplianceChecker()
    
    # Test question
    test_question = "Does the contract require multi-factor authentication (MFA) for all privileged accounts?"
    
    result = checker.check_compliance(test_question)
    
    print("\n" + "="*60)
    print("COMPLIANCE CHECK RESULT")
    print("="*60)
    print(checker.format_result({"question": test_question, **result}))
    
    print("\n" + "="*60)
    print("JSON OUTPUT")
    print("="*60)
    print(checker.get_json_result(result))


if __name__ == "__main__":
    main()
