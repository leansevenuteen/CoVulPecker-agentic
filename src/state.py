"""State schema for LangGraph workflow."""
from typing import Optional, TypedDict, Annotated
from operator import add

from src.models import (
    ClassificationResult,
    DetectionResult,
    ReasoningResult,
    CriticResult,
    VerificationResult,
)


class GraphState(TypedDict):
    """State that flows through the vulnerability detection graph."""
    
    # Input
    source_code: str
    
    # Stage 1: Classification
    classification: Optional[ClassificationResult]
    
    # Stage 2: Detection
    detection: Optional[DetectionResult]
    
    # Stage 3: Reasoning
    reasoning: Optional[ReasoningResult]
    
    # Stage 4: Critic
    critic: Optional[CriticResult]
    critic_feedback_history: Annotated[list[str], add]  # Accumulates feedback
    
    # Stage 5: Verification
    verification: Optional[VerificationResult]
    
    # Control flow
    iteration_count: int
    max_iterations: int
    current_stage: str
    is_complete: bool

