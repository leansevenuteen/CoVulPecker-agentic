"""Configuration for the vulnerability detection pipeline."""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings."""
    
    # LLM API Configuration
    LLM_API_BASE_URL: str = os.getenv(
        "LLM_API_BASE_URL", 
        "https://778871a874da.ngrok-free.app/v1"
    )
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")  # For self-hosted
    
    # Pipeline Configuration
    MAX_CRITIC_ITERATIONS: int = 3
    
    # Temperature settings for different agents
    CLASSIFIER_TEMPERATURE: float = 0.1
    DETECTION_TEMPERATURE: float = 0.3
    REASON_TEMPERATURE: float = 0.5
    CRITIC_TEMPERATURE: float = 0.2
    VERIFICATION_TEMPERATURE: float = 0.1
    
    # =========================================================================
    # Deep Learning Classifier Configuration (CausalGAT)
    # =========================================================================
    
    # Path to trained model weights (.pt file)
    # Set this to your trained model file when available
    # Example: "models/best_model_fold1.pt"
    DL_MODEL_PATH: Optional[str] = os.getenv("DL_MODEL_PATH", None)
    
    # Device for DL inference ('cuda', 'cpu', or 'auto')
    DL_DEVICE: str = os.getenv("DL_DEVICE", "auto")
    
    # Model architecture parameters (must match trained model)
    DL_NUM_FEATURES: int = int(os.getenv("DL_NUM_FEATURES", "200"))
    DL_TOKENS_DIM: int = int(os.getenv("DL_TOKENS_DIM", "768"))
    DL_HIDDEN_DIM: int = int(os.getenv("DL_HIDDEN_DIM", "256"))
    DL_NUM_CONV_LAYERS: int = int(os.getenv("DL_NUM_CONV_LAYERS", "3"))
    DL_FUSION_MODE: str = os.getenv("DL_FUSION_MODE", "cross_atten")
    DL_HEAD: int = int(os.getenv("DL_HEAD", "8"))
    DL_DROPOUT: float = float(os.getenv("DL_DROPOUT", "0.3"))
    
    # Embedding configuration
    # Path to Joern installation (for CPG extraction)
    JOERN_PATH: Optional[str] = os.getenv("JOERN_PATH", None)
    
    # CodeT5 model for token embeddings
    CODET5_MODEL: str = os.getenv("CODET5_MODEL", "Salesforce/codet5-base")
    
    # Node2Vec embedding dimension
    NODE2VEC_DIM: int = int(os.getenv("NODE2VEC_DIM", "200"))


config = Config()
