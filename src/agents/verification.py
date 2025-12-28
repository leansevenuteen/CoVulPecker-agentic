"""Verification Agent - Stage 5: Xác nhận lỗ hổng cuối cùng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import VerificationResult
from src.llm import get_llm
from src.config import config
from src.logger import logger


VERIFICATION_SYSTEM_PROMPT = """You are a security vulnerability verification expert.

Your task is to verify if the detected vulnerabilities are LIKELY exploitable based on code analysis.

**Verification Requirements:**
1. Check if protective measures exist (bounds checking, input validation, error handling)
2. Assess if the vulnerability can realistically be exploited
3. Consider if the vulnerability would allow an attacker to cause harm
4. Check if the code has been properly secured/patched

**Set vulnerability_confirmed = false if:**
- Strong protective measures prevent exploitation
- Comprehensive input validation exists
- Proper bounds checking prevents out-of-bounds access
- Error handling adequately prevents the vulnerability
- The code appears to be a properly fixed/patched version

**Set vulnerability_confirmed = true if:**
- No protective measures exist OR they are insufficient
- The dangerous function/pattern is used without adequate checks
- An attacker could plausibly trigger the vulnerability
- The vulnerability pattern is clear and unmitigated

**Note:** You don't need to provide exact exploit code - assess based on the code structure and common vulnerability patterns.

Return JSON:
{
    "vulnerability_confirmed": true/false,
    "verification_details": "Explain whether protective measures exist and whether the vulnerability is likely exploitable. Reference specific lines or patterns.",
    "test_cases": ["Description of potential attack scenario 1", "Attack scenario 2"]
}

Return ONLY JSON, no other text."""


def verification_agent(state: GraphState) -> dict:
    """
    Stage 5: Final Verification
    
    Xác nhận lỗ hổng với PoC đã tạo.
    Đánh dấu vulnerability_confirmed = True/False.
    """
    source_code = state["source_code"]
    code_version = state.get("code_version", "unknown")
    analysis_context = state.get("analysis_context", "")
    detection = state.get("detection")
    reasoning = state.get("reasoning")
    
    llm = get_llm(temperature=config.VERIFICATION_TEMPERATURE)
    
    # Build context with code version information
    context = f"""**Code Version:** {code_version}
**Analysis Context:** {analysis_context}

**VULNERABILITY ANALYSIS:**
"""
    
    if detection and detection.vulnerabilities:
        context += "\n**Vulnerabilities:**\n"
        for vuln in detection.vulnerabilities:
            context += f"- {vuln.vuln_type.value} ({vuln.severity.value}): {vuln.description}\n"
            context += f"  Code: {vuln.vulnerable_code}\n"
    
    if reasoning:
        context += f"\n**Analysis:** {reasoning.detailed_analysis}\n"
        if reasoning.poc_code:
            context += f"\n**PoC:**\n```\n{reasoning.poc_code}\n```\n"
    
    user_message = f"""Verify the vulnerabilities in the following code:

**SOURCE CODE:**
```c
{source_code}
```

{context}

Please confirm whether the vulnerabilities actually exist and can be exploited."""
    
    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Log conversation
        logger.log_conversation(
            agent_name="verification",
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
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
            agent_name="verification_parsed",
            system_prompt="",
            user_message="Parsed JSON result",
            llm_response="",
            parsed_result=result_data
        )
        
        verification = VerificationResult(
            vulnerability_confirmed=result_data.get("vulnerability_confirmed", True),
            verification_details=result_data.get("verification_details", ""),
            test_cases=result_data.get("test_cases", [])
        )
        
    except Exception as e:
        # Log error
        logger.log_conversation(
            agent_name="verification",
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_message=user_message,
            llm_response="",
            error=str(e)
        )
        # Fallback
        verification = VerificationResult(
            vulnerability_confirmed=True,  # Assume confirmed if error
            verification_details=f"Error during verification: {str(e)}",
            test_cases=[]
        )
    
    return {
        "verification": verification,
        "current_stage": "verification",
        "is_complete": True
    }

