import operator
import yaml
from typing import Annotated, List, Optional, Union, Literal, Dict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from llm_client import StructuredLLMClient
from models import (
    SessionState, RawTextExtraction, ValidationResult, 
    MissingTechnicalDetail, TechAnswerValidation, AgileArtifact
)

# Load prompts from configuration
with open("prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)

class AgentState(BaseModel):
    session: SessionState = Field(default_factory=SessionState)
    raw_input: str = ""
    current_field: Optional[str] = None
    pending_questions: List[str] = Field(default_factory=list)
    phase1_retries: int = 0
    feedback_retries: int = 0
    final_artifact: Optional[AgileArtifact] = None
    is_complete: bool = False

class JiraStatelessAgent:
    def __init__(self):
        self.llm = StructuredLLMClient()

    def phase0_extractor(self, state: AgentState) -> Dict:
        system_msg = PROMPTS["system_personas"]["extractor"]
        user_msg = PROMPTS["user_prompts"]["extract_raw_text"].format(raw_text=state.raw_input)
        
        extraction: RawTextExtraction = self.llm.query(
            system_prompt=system_msg,
            user_prompt=user_msg,
            response_model=RawTextExtraction
        )
        
        updated_session = state.session
        updated_session.who = extraction.who.evidence if extraction.who.identified else None
        updated_session.what = extraction.what.evidence if extraction.what.identified else None
        updated_session.why = extraction.why.evidence if extraction.why.identified else None
        updated_session.ac_evidence = extraction.ac_evidence
        
        return {"session": updated_session}

    def phase1_field_selector(self, state: AgentState) -> Dict:
        if not state.session.who:
            return {"current_field": "who"}
        if not state.session.what:
            return {"current_field": "what"}
        if not state.session.why:
            return {"current_field": "why"}
        return {"current_field": None}

    def phase2_tech_lead(self, state: AgentState) -> Dict:
        system_msg = PROMPTS["system_personas"]["tech_lead"]
        user_msg = PROMPTS["user_prompts"]["generate_dynamic_questions"].format(
            what_statement=state.session.what
        )
        
        tech_analysis: MissingTechnicalDetail = self.llm.query(
            system_prompt=system_msg,
            user_prompt=user_msg,
            response_model=MissingTechnicalDetail
        )
        return {"pending_questions": tech_analysis.questions}

    def phase3_agile_synthesis(self, state: AgentState) -> Dict:
        system_msg = PROMPTS["system_personas"]["agile_coach"]
        user_msg = PROMPTS["user_prompts"]["generate_agile_artifact"].format(
            who=state.session.who,
            what=state.session.what,
            why=state.session.why,
            tech_notes=", ".join(state.session.tech_notes),
            ac_evidence=state.session.ac_evidence or "None"
        )
        
        artifact: AgileArtifact = self.llm.query(
            system_prompt=system_msg,
            user_prompt=user_msg,
            response_model=AgileArtifact
        )
        return {"final_artifact": artifact}

# Routing Logic
def route_post_extraction(state: AgentState):
    s = state.session
    if all([s.who, s.what, s.why]):
        return "tech_grooming"
    return "field_validator"

def route_field_logic(state: AgentState):
    if state.phase1_retries >= 3:
        return "terminate_failure"
    if state.current_field is None:
        return "tech_grooming"
    return "human_input_required"

def route_tech_questions(state: AgentState):
    if not state.pending_questions:
        return "synthesis"
    return "ask_tech_question"

# Graph Construction
workflow = StateGraph(AgentState)
agent = JiraStatelessAgent()

workflow.add_node("extractor", agent.phase0_extractor)
workflow.add_node("field_validator", agent.phase1_field_selector)
workflow.add_node("tech_grooming", agent.phase2_tech_lead)
workflow.add_node("synthesis", agent.phase3_agile_synthesis)

workflow.set_entry_point("extractor")

workflow.add_conditional_edges(
    "extractor",
    route_post_extraction,
    {
        "tech_grooming": "tech_grooming",
        "field_validator": "field_validator"
    }
)

workflow.add_conditional_edges(
    "field_validator",
    route_field_logic,
    {
        "tech_grooming": "tech_grooming",
        "human_input_required": "field_validator",
        "terminate_failure": END
    }
)

workflow.add_conditional_edges(
    "tech_grooming",
    route_tech_questions,
    {
        "synthesis": "synthesis",
        "ask_tech_question": "tech_grooming"
    }
)

workflow.add_edge("synthesis", END)

app = workflow.compile()
