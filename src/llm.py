"""LLM client configuration."""
from langchain_openai import ChatOpenAI
from src.config import config


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Get configured LLM instance."""
    return ChatOpenAI(
        base_url=config.LLM_API_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL_NAME,
        temperature=temperature,
    )

