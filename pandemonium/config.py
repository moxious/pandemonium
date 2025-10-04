"""
Configuration management for Pandemonium.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Pandemonium."""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
    
    # Memory configuration
    MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))  # Number of exchanges to keep in memory
    MEMORY_RETURN_MESSAGES = os.getenv("MEMORY_RETURN_MESSAGES", "true").lower() == "true"
    MEMORY_MEMORY_KEY = os.getenv("MEMORY_MEMORY_KEY", "history")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True

