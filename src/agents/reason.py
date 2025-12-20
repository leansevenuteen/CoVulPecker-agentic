"""Reason Agent - Stage 3: Tạo lý luận chi tiết về lỗ hổng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import ReasoningResult
from src.llm import get_llm
from src.config import config
from src.logger import logger


REASON_SYSTEM_PROMPT = """You are a security expert tasked with providing detailed explanations about security vulnerabilities.

Based on the vulnerability analysis results, you need to:

1. **Detailed Analysis (detailed_analysis)**: Clearly explain why the code is vulnerable and how the vulnerability works.

2. **Exploitation Scenarios (exploitation_scenarios)**: List the ways an attacker could exploit this vulnerability.

3. **Remediation Suggestions (remediation_suggestions)**: Specific steps to fix the vulnerability.

4. **Proof of Concept (poc_code)**: Code demonstrating how to exploit the vulnerability (if applicable).

Return the result in JSON format:
{
    "detailed_analysis": "Detailed analysis...",
    "exploitation_scenarios": ["Scenario 1", "Scenario 2"],
    "remediation_suggestions": ["Fix 1", "Fix 2"],
    "poc_code": "// PoC code here..."
}

Return ONLY JSON, no other text."""


def reason_agent(state: GraphState) -> dict:
    """
    Stage 3: Reason Agent
    
    Tạo lý luận chi tiết về lỗ hổng:
    - Giải thích tại sao code vulnerable
    - Đề xuất cách khai thác
    - Đề xuất cách khắc phục
    - Tạo PoC
    """
    source_code = state["source_code"]
    detection = state.get("detection")
    
    llm = get_llm(temperature=config.REASON_TEMPERATURE)
    
    # Build context from detection result
    detection_summary = ""
    if detection:
        detection_summary = f"\n\n**Detection Results:**\n{detection.summary}\n\n"
        if detection.vulnerabilities:
            detection_summary += "**Detected Vulnerabilities:**\n"
            for i, vuln in enumerate(detection.vulnerabilities, 1):
                detection_summary += f"{i}. {vuln.vuln_type.value} ({vuln.severity.value}) - Line {vuln.line_number}: {vuln.description}\n"
    
    user_message = f"""Provide detailed analysis of the vulnerabilities in the following code:

```c
{source_code}
```
{detection_summary}

Please explain in detail and suggest exploitation methods as well as remediation."""
    
    messages = [
        SystemMessage(content=REASON_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Log conversation
        logger.log_conversation(
            agent_name="reason",
            system_prompt=REASON_SYSTEM_PROMPT,
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
            agent_name="reason_parsed",
            system_prompt="",
            user_message="Parsed JSON result",
            llm_response="",
            parsed_result=result_data
        )
        
        # Handle case where detailed_analysis might be a dict instead of string
        detailed_analysis = result_data.get("detailed_analysis", "")
        if isinstance(detailed_analysis, dict):
            # Convert dict to formatted string
            detailed_analysis = json.dumps(detailed_analysis, indent=2, ensure_ascii=False)
        
        # Handle exploitation_scenarios - ensure it's a list of strings
        exploitation_scenarios = result_data.get("exploitation_scenarios", [])
        if isinstance(exploitation_scenarios, dict):
            exploitation_scenarios = [f"{k}: {v}" for k, v in exploitation_scenarios.items()]
        elif not isinstance(exploitation_scenarios, list):
            exploitation_scenarios = [str(exploitation_scenarios)]
        
        # Handle remediation_suggestions - ensure it's a list of strings  
        remediation_suggestions = result_data.get("remediation_suggestions", [])
        if isinstance(remediation_suggestions, dict):
            remediation_suggestions = [f"{k}: {v}" for k, v in remediation_suggestions.items()]
        elif not isinstance(remediation_suggestions, list):
            remediation_suggestions = [str(remediation_suggestions)]
        
        # Handle poc_code
        poc_code = result_data.get("poc_code")
        if isinstance(poc_code, dict):
            poc_code = json.dumps(poc_code, indent=2, ensure_ascii=False)
        
        reasoning = ReasoningResult(
            detailed_analysis=str(detailed_analysis),
            exploitation_scenarios=[str(s) for s in exploitation_scenarios],
            remediation_suggestions=[str(s) for s in remediation_suggestions],
            poc_code=str(poc_code) if poc_code else None
        )
        
    except Exception as e:
        # Log error
        logger.log_conversation(
            agent_name="reason",
            system_prompt=REASON_SYSTEM_PROMPT,
            user_message=user_message,
            llm_response="",
            error=str(e)
        )
        # Fallback
        reasoning = ReasoningResult(
            detailed_analysis=f"Error during reasoning: {str(e)}",
            exploitation_scenarios=[],
            remediation_suggestions=[],
            poc_code=None
        )
    
    return {
        "reasoning": reasoning,
        "current_stage": "reason"
    }

