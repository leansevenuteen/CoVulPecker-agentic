"""Detection Agent - Stage 2: Phân tích chi tiết các lỗ hổng."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import GraphState
from src.models import DetectionResult, VulnerabilityDetail, VulnerabilityType, Severity
from src.llm import get_llm
from src.config import config
from src.logger import logger


DETECTION_SYSTEM_PROMPT = """You are a software security expert specializing in analyzing vulnerabilities in C/C++ code.

Your task is to analyze source code and identify POTENTIAL security vulnerabilities that COULD be exploited.

**CRITICAL INSTRUCTIONS:**
- Report vulnerabilities if dangerous functions are used WITHOUT adequate protective measures
- Check for protective measures: bounds checking, input validation, null checks, size limits
- If protective measures are MISSING or INSUFFICIENT, report the vulnerability
- Be thorough - it's better to flag a potential issue than miss a real vulnerability

**Examples of SAFE code (NOT vulnerable):**
- strcpy() with validated buffer size: if (strlen(src) < BUFFER_SIZE) strcpy(dst, src);
- malloc() with null check: ptr = malloc(size); if (!ptr) return NULL;
- Array access with bounds check: if (index >= 0 && index < array_len) array[index] = value;

**Examples of VULNERABLE code (REPORT THESE):**
- strcpy() without size validation: strcpy(buffer, user_input);
- malloc() without null check before dereferencing: ptr = malloc(size); *ptr = value;
- Array access without bounds check: array[user_index] = value;
- Use-after-free: free(ptr); /* ... */ *ptr = value;
- Memory leak: ptr = malloc(size); /* no free() */
- Null pointer dereference: ptr->field without checking if ptr is NULL

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
    
    DL Classifier result is provided as metadata/hint, not as a gate.
    The LLM agent performs its own independent analysis.
    """
    source_code = state["source_code"]
    code_version = state.get("code_version", "unknown")
    analysis_context = state.get("analysis_context", "")
    classification = state.get("classification")  # Get DL classifier result as hint
    feedback_history = state.get("critic_feedback_history", [])
    
    llm = get_llm(temperature=config.DETECTION_TEMPERATURE)
    
    # Build prompt with code version context and DL classifier hint
    user_message = f"""Analyze security vulnerabilities in the following code:

**Code Version:** {code_version}
**Analysis Instructions:** {analysis_context}

**SOURCE CODE:**
```c
{source_code}
```"""
    
    # Add DL classifier hint as metadata (not as instruction)
    if classification:
        dl_hint = f"DL Model Hint: {classification.label} (confidence: {classification.confidence:.2%})"
        if classification.detected_patterns:
            patterns = ", ".join(classification.detected_patterns[:3])
            dl_hint += f" | Patterns: {patterns}"
        user_message += f"\n\n📊 {dl_hint}"
        user_message += "\nNote: This is just a hint from a preliminary classifier. Perform your own thorough independent analysis."
    
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

