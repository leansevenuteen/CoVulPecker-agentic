"""
Classifier Agent - Stage 1: Phân loại code có lỗ hổng hay không.

This agent uses a CausalGAT deep learning model for vulnerability classification.
The pipeline:
    1. Receive source code
    2. Call DL Classifier Service (as a tool/function)
    3. Transform source code to embeddings (handled by service)
    4. Run model inference
    5. Return classification result

When the model is not available, it falls back to auto-labeling as vulnerable.
"""

from src.state import GraphState
from src.models import ClassificationResult
from src.config import config


def get_dl_classifier():
    """
    Get the DL classifier service instance.
    
    This acts as a "tool call" that the classifier agent uses.
    The service handles:
    - Loading the trained CausalGAT model
    - Transforming source code to embeddings
    - Running inference
    
    Returns:
        DLClassifierService instance
    """
    try:
        from src.dl_classifier import DLClassifierService
        from src.dl_classifier.service import get_classifier_service
        
        # Get or create service with configured model path and parameters
        service = get_classifier_service(
            model_path=config.DL_MODEL_PATH,
            fusion_mode=config.DL_FUSION_MODE,
            hidden_dim=config.DL_HIDDEN_DIM,
            num_conv_layers=config.DL_NUM_CONV_LAYERS,
            head=config.DL_HEAD,
            dropout=config.DL_DROPOUT
        )
        return service
    except ImportError as e:
        print(f"⚠️ DL Classifier not available: {e}")
        return None


def classifier_agent(state: GraphState) -> dict:
    """
    Stage 1: Classifier Agent
    
    Uses CausalGAT deep learning model to classify source code as
    vulnerable or clean.
    
    Pipeline:
    1. Get source code from state
    2. Call DL classifier service (tool call)
    3. The service transforms code -> graph -> embeddings -> prediction
    4. Return classification result
    
    When model is not available, auto-labels as vulnerable per pipeline.md.
    """
    source_code = state["source_code"]
    
    # Get the DL classifier service (this is the "tool call")
    classifier_service = get_dl_classifier()
    
    if classifier_service is not None:
        # Call the model through the service
        print(f"🔍 Calling DL Classifier (mode: {classifier_service.mode.value})")
        
        dl_result = classifier_service.classify(source_code)
        
        # Log model status
        if dl_result.model_available:
            print(f"   ✓ Model prediction: {dl_result.label} ({dl_result.confidence:.2%} confidence)")
        else:
            print(f"   ⚠️ Model not loaded, using placeholder (auto-label: vulnerable)")
        
        classification = ClassificationResult(
            label=dl_result.label,
            confidence=dl_result.confidence,
            detected_patterns=dl_result.detected_patterns
        )
    else:
        # Fallback: DL module not available
        print("⚠️ DL Classifier module not available, using heuristic fallback")
        
        # Detect patterns using basic heuristics
        detected_patterns = _detect_patterns_heuristic(source_code)
        
        classification = ClassificationResult(
            label="vulnerable",  # Auto-label as vulnerable
            confidence=0.95,     # High confidence (placeholder)
            detected_patterns=detected_patterns
        )
    
    return {
        "classification": classification,
        "current_stage": "classifier"
    }


def _detect_patterns_heuristic(source_code: str) -> list:
    """
    Fallback heuristic pattern detection.
    
    Used when DL classifier is not available.
    """
    detected_patterns = []
    
    dangerous_functions = {
        "strcpy": "Buffer Overflow",
        "sprintf": "Buffer Overflow / Format String",
        "gets": "Buffer Overflow", 
        "scanf": "Buffer Overflow",
        "printf": "Format String (if user input)",
        "free": "Use After Free / Double Free",
        "malloc": "Memory issues",
        "system": "Command Injection",
        "exec": "Command Injection",
        "eval": "Code Injection",
    }
    
    for func, vuln_type in dangerous_functions.items():
        if func in source_code:
            detected_patterns.append(f"{func} -> Potential {vuln_type}")
    
    return detected_patterns
