"""Data models for the vulnerability detection pipeline."""
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class VulnerabilityType(str, Enum):
    BUFFER_OVERFLOW = "buffer_overflow"
    FORMAT_STRING = "format_string"
    USE_AFTER_FREE = "use_after_free"
    INTEGER_OVERFLOW = "integer_overflow"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    MEMORY_LEAK = "memory_leak"
    NULL_POINTER = "null_pointer"
    RACE_CONDITION = "race_condition"
    OTHER = "other"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ClassificationResult(BaseModel):
    """Result from the Classifier Agent."""
    label: str = Field(description="vulnerable or safe")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    detected_patterns: List[str] = Field(default_factory=list, description="Detected dangerous patterns")


class VulnerabilityDetail(BaseModel):
    """Details about a specific vulnerability."""
    vuln_type: VulnerabilityType = Field(description="Type of vulnerability")
    severity: Severity = Field(description="Severity level")
    line_number: Optional[int] = Field(default=None, description="Line number where vulnerability exists")
    function_name: Optional[str] = Field(default=None, description="Function containing the vulnerability")
    vulnerable_code: str = Field(description="The vulnerable code snippet")
    description: str = Field(description="Description of the vulnerability")


class DetectionResult(BaseModel):
    """Result from the Detection Agent."""
    vulnerabilities: List[VulnerabilityDetail] = Field(default_factory=list)
    summary: str = Field(description="Summary of detected vulnerabilities")


class ReasoningResult(BaseModel):
    """Result from the Reason Agent."""
    detailed_analysis: str = Field(description="Detailed analysis of vulnerabilities")
    exploitation_scenarios: List[str] = Field(default_factory=list, description="Possible exploitation scenarios")
    remediation_suggestions: List[str] = Field(default_factory=list, description="How to fix the vulnerabilities")
    poc_code: Optional[str] = Field(default=None, description="Proof of Concept exploit code")


class CriticDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class CriticResult(BaseModel):
    """Result from the Critic Agent."""
    decision: CriticDecision = Field(description="Approved or rejected")
    feedback: Optional[str] = Field(default=None, description="Feedback if rejected")
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality score of the analysis")
    issues_found: List[str] = Field(default_factory=list, description="Issues found in the analysis")


class VerificationResult(BaseModel):
    """Result from the Final Verification stage."""
    vulnerability_confirmed: bool = Field(description="Whether vulnerability is confirmed")
    verification_details: str = Field(description="Details of verification")
    test_cases: List[str] = Field(default_factory=list, description="Test cases used")


class FinalReport(BaseModel):
    """Final output report."""
    source_code: str = Field(description="Original source code")
    classification: ClassificationResult
    detection: DetectionResult
    reasoning: ReasoningResult
    critic_review: CriticResult
    verification: VerificationResult
    total_iterations: int = Field(default=1, description="Number of detection-critic iterations")

