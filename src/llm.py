"""LLM client configuration."""
from langchain_openai import ChatOpenAI
from src.config import config


def get_llm(temperature: float = 0.3, timeout: int = 120) -> ChatOpenAI:
    """
    Get configured LLM instance with timeout.
    
    Args:
        temperature: Sampling temperature
        timeout: Request timeout in seconds (default: 120s)
    """
    return ChatOpenAI(
        base_url=config.LLM_API_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL_NAME,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
    )

