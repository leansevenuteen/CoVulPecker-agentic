"""LangGraph workflow for vulnerability detection pipeline."""
from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.models import CriticDecision
from src.config import config
from src.agents import (
    classifier_agent,
    detection_agent,
    reason_agent,
    critic_agent,
    verification_agent,
)


def should_continue_after_classifier(state: GraphState) -> str:
    """
    Quyết định sau Classifier:
    - Nếu label = "vulnerable" -> tiếp tục detection
    - Nếu label = "safe" -> kết thúc
    """
    classification = state.get("classification")
    
    if classification and classification.label == "vulnerable":
        return "detection"
    else:
        return "end"


def should_retry_detection(state: GraphState) -> str:
    """
    Quyết định sau Critic:
    - Nếu APPROVED -> tiếp tục verification
    - Nếu REJECTED và còn iteration -> quay lại detection
    - Nếu REJECTED nhưng hết iteration -> tiếp tục verification
    """
    critic = state.get("critic")
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", config.MAX_CRITIC_ITERATIONS)
    
    if critic:
        if critic.decision == CriticDecision.APPROVED:
            return "verification"
        elif iteration_count < max_iterations:
            return "detection"  # Retry with feedback
        else:
            return "verification"  # Max retries reached
    
    return "verification"


def create_vulnerability_detection_graph() -> StateGraph:
    """
    Tạo LangGraph workflow cho vulnerability detection.
    
    Flow:
    1. Classifier -> phân loại code
    2. Detection -> phát hiện lỗ hổng  
    3. Reason -> giải thích chi tiết
    4. Critic -> đánh giá (có thể loop lại Detection)
    5. Verification -> xác nhận cuối cùng
    """
    
    # Khởi tạo graph với state schema
    workflow = StateGraph(GraphState)
    
    # Thêm các nodes (agents)
    workflow.add_node("classifier", classifier_agent)
    workflow.add_node("detection", detection_agent)
    workflow.add_node("reason", reason_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("verification", verification_agent)
    
    # Đặt entry point
    workflow.set_entry_point("classifier")
    
    # Thêm conditional edge sau classifier
    workflow.add_conditional_edges(
        "classifier",
        should_continue_after_classifier,
        {
            "detection": "detection",
            "end": END
        }
    )
    
    # Detection -> Reason (always)
    workflow.add_edge("detection", "reason")
    
    # Reason -> Critic (always)
    workflow.add_edge("reason", "critic")
    
    # Critic -> Detection (retry) hoặc Verification (approved)
    workflow.add_conditional_edges(
        "critic",
        should_retry_detection,
        {
            "detection": "detection",
            "verification": "verification"
        }
    )
    
    # Verification -> END
    workflow.add_edge("verification", END)
    
    return workflow


def compile_graph():
    """Compile graph để sẵn sàng chạy."""
    workflow = create_vulnerability_detection_graph()
    return workflow.compile()


# Pre-compiled graph instance
vulnerability_detector = compile_graph()

