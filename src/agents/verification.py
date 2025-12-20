"""Verification Agent - Stage 5: Xác nhận lỗ hổng cuối cùng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import VerificationResult
from src.llm import get_llm
from src.config import config
from src.logger import logger


VERIFICATION_SYSTEM_PROMPT = """You are a security vulnerability verification expert.

Your task is to confirm the detected vulnerabilities:

1. Check if the vulnerabilities actually exist
2. Verify if the PoC works
3. Assess the actual danger level

Return JSON:
{
    "vulnerability_confirmed": true/false,
    "verification_details": "Verification details...",
    "test_cases": ["Test case 1", "Test case 2"]
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

