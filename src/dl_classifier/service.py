"""
Deep Learning Classifier Service.

This service orchestrates the vulnerability classification pipeline:
1. Receive source code
2. Transform to embeddings (code -> graph -> matrix)
3. Run CausalGAT model inference
4. Return classification result

The service can work in different modes:
- PLACEHOLDER: Uses placeholder embeddings and random predictions (for testing)
- INFERENCE: Uses trained model for real predictions
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

# Check for dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .embeddings import CodeEmbedder, PlaceholderEmbedder, get_default_embedder


class ClassifierMode(Enum):
    """Operating mode for the classifier service."""
    PLACEHOLDER = "placeholder"  # Use placeholder embeddings, auto-label as vulnerable
    INFERENCE = "inference"      # Use real model for inference


@dataclass
class DLClassificationResult:
    """Result from the DL classifier."""
    label: str                    # "vulnerable" or "clean"
    confidence: float             # Confidence score [0, 1]
    predicted_class: int          # Raw class prediction (0=clean, 1=vulnerable)
    model_available: bool         # Whether a real model was used
    detected_patterns: List[str]  # Detected patterns (from heuristics)
    
    def is_vulnerable(self) -> bool:
        return self.label == "vulnerable"


class DLClassifierService:
    """
    Deep Learning Classifier Service for Vulnerability Detection.
    
    This service wraps the CausalGAT model and provides a simple interface
    for classifying source code as vulnerable or clean.
    
    Usage:
        # Initialize service
        service = DLClassifierService()
        
        # Load trained model (when available)
        service.load_model("path/to/model.pt")
        
        # Classify source code
        result = service.classify(source_code)
        if result.is_vulnerable():
            print(f"Vulnerable with {result.confidence:.2%} confidence")
    """
    
    # Class label mapping
    LABEL_CLEAN = 0
    LABEL_VULNERABLE = 1
    LABEL_NAMES = {0: "clean", 1: "vulnerable"}
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 embedder: Optional[CodeEmbedder] = None,
                 device: Optional[str] = None,
                 num_features: int = 200,
                 tokens_dim: int = 768,
                 hidden_dim: int = 256,
                 num_conv_layers: int = 3,
                 fusion_mode: str = "cross_atten",
                 head: int = 8,
                 dropout: float = 0.3):
        """
        Initialize the classifier service.
        
        Args:
            model_path: Path to trained model weights (.pt file)
            embedder: Code embedder instance (uses default if None)
            device: Device for inference ('cuda', 'cpu', or None for auto)
            num_features: Dimension of graph node features
            tokens_dim: Dimension of token embeddings
            hidden_dim: Hidden layer dimension
            num_conv_layers: Number of GAT layers
            fusion_mode: Feature fusion mode ('concat', 'gated', 'cross_atten')
            head: Number of attention heads
            dropout: Dropout probability
        """
        self.model_path = model_path
        self.embedder = embedder or get_default_embedder()
        
        # Model architecture params
        self.num_features = num_features
        self.tokens_dim = tokens_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.fusion_mode = fusion_mode
        self.head = head
        self.dropout = dropout
        
        # Set device
        if device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Model instance
        self._model = None
        self._mode = ClassifierMode.PLACEHOLDER
        
        # Try to load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model weights.
        
        Args:
            model_path: Path to .pt file containing model state dict
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available, cannot load model")
            return False
        
        try:
            # Import model here to avoid circular imports
            from .model import CausalGAT, is_available
            
            if not is_available():
                print("⚠️ torch_geometric not available, cannot load model")
                return False
            
            # Initialize model architecture
            self._model = CausalGAT(
                num_features=self.num_features,
                num_classes=2,  # Binary classification
                tokens_dim=self.tokens_dim,
                hidden_dim=self.hidden_dim,
                num_conv_layers=self.num_conv_layers,
                fusion_mode=self.fusion_mode,
                head=self.head,
                dropout=self.dropout
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model.to(self.device)
            self._model.eval()
            
            self._mode = ClassifierMode.INFERENCE
            self.model_path = model_path
            
            print(f"✅ Model loaded from {model_path}")
            print(f"   Device: {self.device}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            self._model = None
            self._mode = ClassifierMode.PLACEHOLDER
            return False
    
    def classify(self, source_code: str) -> DLClassificationResult:
        """
        Classify source code as vulnerable or clean.
        
        Args:
            source_code: Source code string to classify
            
        Returns:
            DLClassificationResult with classification details
        """
        # Detect patterns using heuristics (always run)
        detected_patterns = self._detect_patterns(source_code)
        
        if self._mode == ClassifierMode.INFERENCE and self._model is not None:
            return self._classify_with_model(source_code, detected_patterns)
        else:
            return self._classify_placeholder(source_code, detected_patterns)
    
    def _classify_with_model(self, source_code: str, patterns: List[str]) -> DLClassificationResult:
        """Run actual model inference."""
        try:
            # Get embeddings
            embeddings = self.embedder.embed(source_code)
            data = embeddings.to_pyg_data(device=self.device)
            
            # Run model
            pred_class, confidence = self._model.predict(data)
            label = self.LABEL_NAMES.get(pred_class, "unknown")
            
            return DLClassificationResult(
                label=label,
                confidence=confidence,
                predicted_class=pred_class,
                model_available=True,
                detected_patterns=patterns
            )
            
        except Exception as e:
            print(f"⚠️ Model inference failed: {e}, falling back to placeholder")
            return self._classify_placeholder(source_code, patterns)
    
    def _classify_placeholder(self, source_code: str, patterns: List[str]) -> DLClassificationResult:
        """
        Placeholder classification when model is not available.
        
        Currently auto-labels as vulnerable (as per pipeline.md requirement).
        """
        # Auto-label as vulnerable (as specified in pipeline.md)
        return DLClassificationResult(
            label="vulnerable",
            confidence=0.95,  # High confidence placeholder
            predicted_class=self.LABEL_VULNERABLE,
            model_available=False,
            detected_patterns=patterns
        )
    
    def _detect_patterns(self, source_code: str) -> List[str]:
        """
        Detect dangerous patterns in source code using heuristics.
        
        This provides additional context beyond the model prediction.
        """
        patterns = []
        
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
            "strcat": "Buffer Overflow",
            "memcpy": "Buffer Overflow",
            "memmove": "Buffer Overflow",
            "realpath": "Path Traversal",
            "popen": "Command Injection",
        }
        
        for func, vuln_type in dangerous_functions.items():
            if func in source_code:
                patterns.append(f"{func} -> Potential {vuln_type}")
        
        return patterns
    
    @property
    def mode(self) -> ClassifierMode:
        """Get current operating mode."""
        return self._mode
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a trained model is loaded."""
        return self._model is not None and self._mode == ClassifierMode.INFERENCE
    
    def get_status(self) -> dict:
        """Get service status information."""
        return {
            "mode": self._mode.value,
            "model_loaded": self.is_model_loaded,
            "model_path": self.model_path,
            "device": self.device,
            "embedder_ready": self.embedder.is_ready() if self.embedder else False,
            "torch_available": TORCH_AVAILABLE,
        }


# Global service instance (lazy initialization)
_service_instance: Optional[DLClassifierService] = None


def get_classifier_service(
    model_path: Optional[str] = None,
    force_reload: bool = False
) -> DLClassifierService:
    """
    Get or create the global classifier service instance.
    
    Args:
        model_path: Path to trained model (optional)
        force_reload: Force recreation of service instance
        
    Returns:
        DLClassifierService instance
    """
    global _service_instance
    
    if _service_instance is None or force_reload:
        _service_instance = DLClassifierService(model_path=model_path)
    elif model_path and not _service_instance.is_model_loaded:
        _service_instance.load_model(model_path)
    
    return _service_instance


