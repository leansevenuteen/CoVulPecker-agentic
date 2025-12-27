"""Detection Agent - Stage 2: Phân tích chi tiết các lỗ hổng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import DetectionResult, VulnerabilityDetail, VulnerabilityType, Severity
from src.llm import get_llm
from src.config import config
from src.logger import logger


DETECTION_SYSTEM_PROMPT = """You are a software security expert specializing in analyzing vulnerabilities in C/C++ code.

Your task is to analyze source code and identify EXPLOITABLE security vulnerabilities.

**CRITICAL INSTRUCTIONS:**
- A vulnerability exists ONLY if it can actually be exploited
- Distinguish between "dangerous function present" vs "dangerous function misused"
- Check for protective measures: bounds checking, input validation, null checks, size limits
- If protective measures exist and are sufficient, DO NOT report a vulnerability

**Examples of SAFE code (NOT vulnerable):**
- strcpy() with validated buffer size: if (strlen(src) < BUFFER_SIZE) strcpy(dst, src);
- malloc() with null check: ptr = malloc(size); if (!ptr) return;
- Array access with bounds check: if (index >= 0 && index < array_len) array[index] = value;

**Examples of VULNERABLE code:**
- strcpy() without size validation: strcpy(buffer, user_input);
- malloc() without null check and dereferencing: ptr = malloc(size); *ptr = value;
- Array access without bounds check: array[user_index] = value;

For each EXPLOITABLE vulnerability, you need to determine:
1. Vulnerability type (vuln_type): buffer_overflow, format_string, use_after_free, integer_overflow, sql_injection, command_injection, xss, path_traversal, memory_leak, null_pointer, race_condition, other
2. Severity level (severity): critical, high, medium, low, info
3. Line number (line_number): Line containing the vulnerability
4. Function name (function_name): Function containing the vulnerability
5. Vulnerable code (vulnerable_code): The code snippet with the vulnerability
6. Description (description): Explain WHY this is exploitable and what protective measures are missing

**If the code has sufficient protective measures, return an empty vulnerabilities array.**

Return the result in JSON format:
{
    "vulnerabilities": [
        {
            "vuln_type": "buffer_overflow",
            "severity": "critical",
            "line_number": 10,
            "function_name": "vulnerable_copy",
            "vulnerable_code": "strcpy(buffer, user_input);",
            "description": "Buffer overflow: strcpy copies user_input without validating length. No bounds checking present. Attacker can provide input longer than buffer size."
        }
    ],
    "summary": "Summary of detected EXPLOITABLE vulnerabilities, or 'No exploitable vulnerabilities found' if code is safe"
}

Return ONLY JSON, no other text."""


def detection_agent(state: GraphState) -> dict:
    """
    Stage 2: Detection Agent
    
    Phân tích chi tiết các lỗ hổng trong code.
    Có thể nhận feedback từ Critic để cải thiện.
    """
    source_code = state["source_code"]
    feedback_history = state.get("critic_feedback_history", [])
    
    llm = get_llm(temperature=config.DETECTION_TEMPERATURE)
    
    # Build prompt with feedback if available
    user_message = f"Analyze security vulnerabilities in the following code:\n\n```c\n{source_code}\n```"
    
    if feedback_history:
        feedback_text = "\n".join([f"- {fb}" for fb in feedback_history])
        user_message += f"\n\n⚠️ FEEDBACK FROM PREVIOUS ANALYSIS:\n{feedback_text}\n\nPlease improve the analysis based on this feedback."
    
    messages = [
        SystemMessage(content=DETECTION_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Log conversation
        logger.log_conversation(
            agent_name="detection",
            system_prompt=DETECTION_SYSTEM_PROMPT,
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
        
        # Update log with parsed result
        logger.log_conversation(
            agent_name="detection_parsed",
            system_prompt="",
            user_message="Parsed JSON result",
            llm_response="",
            parsed_result=result_data
        )
        
        vulnerabilities = []
        for vuln in result_data.get("vulnerabilities", []):
            try:
                vuln_detail = VulnerabilityDetail(
                    vuln_type=VulnerabilityType(vuln.get("vuln_type", "other")),
                    severity=Severity(vuln.get("severity", "medium")),
                    line_number=vuln.get("line_number"),
                    function_name=vuln.get("function_name"),
                    vulnerable_code=vuln.get("vulnerable_code", ""),
                    description=vuln.get("description", "")
                )
                vulnerabilities.append(vuln_detail)
            except Exception:
                continue
        
        detection = DetectionResult(
            vulnerabilities=vulnerabilities,
            summary=result_data.get("summary", "Đã phân tích code")
        )
        
    except Exception as e:
        # Log error
        logger.log_conversation(
            agent_name="detection",
            system_prompt=DETECTION_SYSTEM_PROMPT,
            user_message=user_message,
            llm_response="",
            error=str(e)
        )
        # Fallback nếu LLM gặp lỗi
        detection = DetectionResult(
            vulnerabilities=[],
            summary=f"Error during analysis: {str(e)}"
        )
    
    return {
        "detection": detection,
        "current_stage": "detection"
    }

