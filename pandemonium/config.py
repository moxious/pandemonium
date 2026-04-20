"""
Configuration management for Pandemonium.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Pandemonium."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    # Context window configuration
    CONTEXT_ROUNDS = int(os.getenv("CONTEXT_ROUNDS", "3"))  # Number of past rounds agents can see
    SUMMARY_AFTER_ROUNDS = int(os.getenv("SUMMARY_AFTER_ROUNDS", "3"))  # Start generating summaries after this many rounds

    # Message-based strategy configuration
    CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "20"))  # Sliding window size for message-based strategies
    SUMMARY_AFTER_MESSAGES = int(os.getenv("SUMMARY_AFTER_MESSAGES", "15"))  # Trigger summary after this many messages
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    @classmethod
    def setup_logging(cls):
        """Configure logging for the application."""
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler()  # Output to stdout
            ]
        )
        
        # Set specific log levels for different components
        logging.getLogger("pandemonium").setLevel(getattr(logging, cls.LOG_LEVEL))
        logging.getLogger("pandemonium.agents").setLevel(getattr(logging, cls.LOG_LEVEL))
        logging.getLogger("pandemonium.conversation").setLevel(getattr(logging, cls.LOG_LEVEL))
        
        # Reduce noise from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if cls.DEFAULT_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        if cls.DEFAULT_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Please set it in your .env file.")
        return True


def create_chat_model(provider=None, model=None, temperature=None):
    """Return a LangChain chat model for the given provider."""
    provider = provider or Config.DEFAULT_PROVIDER
    temperature = temperature if temperature is not None else Config.TEMPERATURE

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models. Please set it in your .env file.")
        return ChatAnthropic(
            model=model or Config.ANTHROPIC_MODEL,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models. Please set it in your .env file.")
        return ChatOpenAI(
            model=model or Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'. Supported: openai, anthropic")


class TokenTracker:
    """Tracks cumulative token usage across all LLM calls in a conversation."""

    def __init__(self, warning_threshold: int = 120_000):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.warning_threshold = warning_threshold
        self._warned = False
        self._logger = logging.getLogger("pandemonium.tokens")

    def track(self, input_tokens: int, output_tokens: int):
        """Record token usage from an LLM call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        total = self.total_tokens
        if total >= self.warning_threshold and not self._warned:
            self._warned = True
            self._logger.warning(
                f"Token usage warning: {total:,} tokens used "
                f"(threshold: {self.warning_threshold:,}). "
                f"Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,}"
            )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

