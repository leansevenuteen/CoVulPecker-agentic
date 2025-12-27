"""Verification Agent - Stage 5: Xác nhận lỗ hổng cuối cùng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import VerificationResult
from src.llm import get_llm
from src.config import config
from src.logger import logger


VERIFICATION_SYSTEM_PROMPT = """You are a security vulnerability verification expert.

Your task is to rigorously verify if the detected vulnerabilities are ACTUALLY EXPLOITABLE.

**Verification Requirements:**
1. Check if protective measures exist (bounds checking, input validation, error handling)
2. Verify if the PoC can actually trigger the vulnerability despite any protections
3. Confirm the vulnerability is exploitable with concrete attack scenarios
4. Check if the code has been properly secured/patched

**Set vulnerability_confirmed = false if:**
- Protective measures prevent exploitation
- Input validation exists and is sufficient
- Bounds checking prevents out-of-bounds access
- Error handling prevents the vulnerability
- The code appears to be a fixed/patched version

**Set vulnerability_confirmed = true ONLY if:**
- You can provide specific input values that trigger the vulnerability
- No protective measures exist or they are insufficient
- The vulnerability is demonstrably exploitable

Return JSON:
{
    "vulnerability_confirmed": true/false,
    "verification_details": "Explain whether protective measures exist and whether the vulnerability is actually exploitable. Be specific about what input would trigger it, or why it cannot be exploited.",
    "test_cases": ["Specific test case 1 with input values", "Test case 2"]
}

Return ONLY JSON, no other text."""


def verification_agent(state: GraphState) -> dict:
    """
    Stage 5: Final Verification
    
    Xác nhận lỗ hổng với PoC đã tạo.
    Đánh dấu vulnerability_confirmed = True/False.
    """
    source_code = state["source_code"]
    detection = state.get("detection")
    reasoning = state.get("reasoning")
    
    llm = get_llm(temperature=config.VERIFICATION_TEMPERATURE)
    
    # Build context
    context = "**VULNERABILITY ANALYSIS:**\n\n"
    
    if detection and detection.vulnerabilities:
        context += "**Vulnerabilities:**\n"
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

