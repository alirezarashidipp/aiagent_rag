# state.py
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Pydantic Models for LLM Client (Structured Outputs)
# ---------------------------------------------------------
class ExtractorOutput(BaseModel):
    who: Optional[str] = Field(None, description="The persona or user role")
    what: Optional[str] = Field(None, description="The core action or feature")
    why: Optional[str] = Field(None, description="The business value or reason")
    ac_evidence: Optional[str] = Field(None, description="Acceptance criteria or constraints")

class ValidatorOutput(BaseModel):
    is_valid: bool
    normalized_value: Optional[str] = Field(None)
    rejection_reason: Optional[str] = Field(None)

class TechQuestionsOutput(BaseModel):
    questions: List[str]

# ---------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------
class WorkflowState(TypedDict):
    # Phase 0: Initial Extraction
    raw_input: str
    who: Optional[str]
    what: Optional[str]
    why: Optional[str]
    ac_evidence: Optional[str]
    
    # Phase 1: JIT Validation
    missing_fields: List[str]
    current_field_target: Optional[str]
    phase1_retries: int
    last_rejection_reason: Optional[str] # Added for rejection tracing
    is_aborted: bool
    abort_reason: Optional[str]
    
    # Phase 2: Tech Grooming
    pending_questions: List[str]
    current_question: Optional[str]
    tech_notes: List[str]
    
    # Phase 3: Synthesis & Feedback
    final_story: Optional[str]
    feedback_retries: int
    is_complete: bool
    feedback_raw: Optional[str]
