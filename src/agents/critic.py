"""Critic Agent - Stage 4: Đánh giá chất lượng phân tích."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import CriticResult, CriticDecision
from src.llm import get_llm
from src.config import config
from src.logger import logger


CRITIC_SYSTEM_PROMPT = """You are an expert in evaluating the quality of security analysis.

Your task is to evaluate the vulnerability analysis results:

**Evaluation Criteria:**
1. Accuracy: Are the identified vulnerabilities correct?
2. Completeness: Are any vulnerabilities missed?
3. Detail Level: Are the descriptions sufficiently detailed?
4. Exploitability: Is the PoC valid?
5. Remediation Suggestions: Are they practical and correct?

**Rules:**
- If quality >= 70%: APPROVED
- If quality < 70%: REJECTED (with specific feedback)

Return JSON:
{
    "decision": "approved" or "rejected",
    "quality_score": 0.0-1.0,
    "feedback": "Feedback if rejected (null if approved)",
    "issues_found": ["Issue 1", "Issue 2"]
}

Return ONLY JSON, no other text."""


def critic_agent(state: GraphState) -> dict:
    """
    Stage 4: Critic Agent
    
    Đánh giá chất lượng phân tích:
    - Nếu APPROVED -> tiếp tục sang verification
    - Nếu REJECTED -> gửi feedback quay lại detection (max 3 lần)
    """
    source_code = state["source_code"]
    detection = state.get("detection")
    reasoning = state.get("reasoning")
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", config.MAX_CRITIC_ITERATIONS)
    
    llm = get_llm(temperature=config.CRITIC_TEMPERATURE)
    
    # Build context
    analysis_summary = "**ANALYSIS RESULTS:**\n\n"
    
    if detection:
        analysis_summary += f"**Detection Summary:** {detection.summary}\n\n"
        if detection.vulnerabilities:
            analysis_summary += "**Vulnerabilities Found:**\n"
            for vuln in detection.vulnerabilities:
                analysis_summary += f"- {vuln.vuln_type.value}: {vuln.description} (Line {vuln.line_number})\n"
    
    if reasoning:
        analysis_summary += f"\n**Detailed Analysis:**\n{reasoning.detailed_analysis}\n"
        if reasoning.exploitation_scenarios:
            analysis_summary += "\n**Exploitation Scenarios:**\n"
            for scenario in reasoning.exploitation_scenarios:
                analysis_summary += f"- {scenario}\n"
        if reasoning.poc_code:
            analysis_summary += f"\n**PoC Code:**\n```\n{reasoning.poc_code}\n```\n"
    
    user_message = f"""Evaluate the quality of the following security analysis:

**SOURCE CODE:**
```c
{source_code}
```

{analysis_summary}

**Analysis Iteration:** {iteration_count}/{max_iterations}

Please evaluate and provide feedback if improvements are needed."""
    
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Log conversation
        logger.log_conversation(
            agent_name="critic",
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_message=user_message,
            llm_response=content
        )
        
        # Parse JSON response
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        result_data = json.loads(content.strip())
        
        # Log parsed result
        logger.log_conversation(
            agent_name="critic_parsed",
            system_prompt="",
            user_message="Parsed JSON result",
            llm_response="",
            parsed_result=result_data
        )
        
        decision_str = result_data.get("decision", "approved").lower()
        decision = CriticDecision.APPROVED if decision_str == "approved" else CriticDecision.REJECTED
        
        # Nếu đã đạt max iterations, force approve
        if iteration_count >= max_iterations:
            decision = CriticDecision.APPROVED
        
        critic_result = CriticResult(
            decision=decision,
            quality_score=result_data.get("quality_score", 0.7),
            feedback=result_data.get("feedback") if decision == CriticDecision.REJECTED else None,
            issues_found=result_data.get("issues_found", [])
        )
        
    except Exception as e:
        # Log error
        logger.log_conversation(
            agent_name="critic",
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_message=user_message,
            llm_response="",
            error=str(e)
        )
        # Fallback - approve để không bị stuck
        critic_result = CriticResult(
            decision=CriticDecision.APPROVED,
            quality_score=0.7,
            feedback=None,
            issues_found=[f"Error during evaluation: {str(e)}"]
        )
    
    # Cập nhật feedback history nếu rejected
    new_feedback = []
    if critic_result.decision == CriticDecision.REJECTED and critic_result.feedback:
        new_feedback = [critic_result.feedback]
    
    return {
        "critic": critic_result,
        "critic_feedback_history": new_feedback,
        "iteration_count": iteration_count + 1 if critic_result.decision == CriticDecision.REJECTED else iteration_count,
        "current_stage": "critic"
    }

