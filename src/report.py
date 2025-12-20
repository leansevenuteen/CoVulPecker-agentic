"""Report generator for vulnerability detection results."""
from src.state import GraphState
from src.models import FinalReport


def generate_report(state: GraphState) -> FinalReport:
    """Generate final report from graph state."""
    return FinalReport(
        source_code=state["source_code"],
        classification=state["classification"],
        detection=state["detection"],
        reasoning=state["reasoning"],
        critic_review=state["critic"],
        verification=state["verification"],
        total_iterations=state.get("iteration_count", 1)
    )


def print_report(report: FinalReport) -> str:
    """Print formatted report."""
    separator = "=" * 80
    sub_separator = "-" * 40
    
    output = []
    output.append(separator)
    output.append("🔒 VULNERABILITY DETECTION REPORT")
    output.append(separator)
    
    # Classification
    output.append("\n📊 STAGE 1: CLASSIFICATION")
    output.append(sub_separator)
    output.append(f"Label: {report.classification.label.upper()}")
    output.append(f"Confidence: {report.classification.confidence:.2%}")
    if report.classification.detected_patterns:
        output.append("Detected Patterns:")
        for pattern in report.classification.detected_patterns:
            output.append(f"  • {pattern}")
    
    # Detection
    output.append("\n🔍 STAGE 2: DETECTION")
    output.append(sub_separator)
    output.append(f"Summary: {report.detection.summary}")
    if report.detection.vulnerabilities:
        output.append("\nVulnerabilities Found:")
        for i, vuln in enumerate(report.detection.vulnerabilities, 1):
            output.append(f"\n  [{i}] {vuln.vuln_type.value.upper()}")
            output.append(f"      Severity: {vuln.severity.value}")
            if vuln.line_number:
                output.append(f"      Line: {vuln.line_number}")
            if vuln.function_name:
                output.append(f"      Function: {vuln.function_name}")
            output.append(f"      Code: {vuln.vulnerable_code}")
            output.append(f"      Description: {vuln.description}")
    
    # Reasoning
    output.append("\n💡 STAGE 3: REASONING")
    output.append(sub_separator)
    output.append(f"Analysis:\n{report.reasoning.detailed_analysis}")
    
    if report.reasoning.exploitation_scenarios:
        output.append("\nExploitation Scenarios:")
        for scenario in report.reasoning.exploitation_scenarios:
            output.append(f"  • {scenario}")
    
    if report.reasoning.remediation_suggestions:
        output.append("\nRemediation Suggestions:")
        for suggestion in report.reasoning.remediation_suggestions:
            output.append(f"  ✓ {suggestion}")
    
    if report.reasoning.poc_code:
        output.append("\nProof of Concept:")
        output.append("```")
        output.append(report.reasoning.poc_code)
        output.append("```")
    
    # Critic Review
    output.append("\n⚖️ STAGE 4: CRITIC REVIEW")
    output.append(sub_separator)
    output.append(f"Decision: {report.critic_review.decision.value.upper()}")
    output.append(f"Quality Score: {report.critic_review.quality_score:.2%}")
    output.append(f"Total Iterations: {report.total_iterations}")
    
    if report.critic_review.issues_found:
        output.append("Issues Found:")
        for issue in report.critic_review.issues_found:
            output.append(f"  ⚠️ {issue}")
    
    # Verification
    output.append("\n✅ STAGE 5: VERIFICATION")
    output.append(sub_separator)
    confirmed_icon = "✅" if report.verification.vulnerability_confirmed else "❌"
    output.append(f"Vulnerability Confirmed: {confirmed_icon} {report.verification.vulnerability_confirmed}")
    output.append(f"Details: {report.verification.verification_details}")
    
    if report.verification.test_cases:
        output.append("\nTest Cases:")
        for tc in report.verification.test_cases:
            output.append(f"  🧪 {tc}")
    
    output.append("\n" + separator)
    output.append("END OF REPORT")
    output.append(separator)
    
    return "\n".join(output)

