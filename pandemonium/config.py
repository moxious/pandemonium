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
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.9"))
    ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "false").lower() == "true"
    ALLOWED_TOOLS = os.getenv("ALLOWED_TOOLS", "web_search,calculator").split(",")

    # Memory configuration
    MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))  # Number of exchanges to keep in memory
    MEMORY_RETURN_MESSAGES = os.getenv("MEMORY_RETURN_MESSAGES", "true").lower() == "true"
    MEMORY_MEMORY_KEY = os.getenv("MEMORY_MEMORY_KEY", "history")
    
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
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True

