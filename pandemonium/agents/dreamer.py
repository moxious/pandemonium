"""
Dreamer agent implementation.
"""

import logging
from .base_agent import BaseAgent


class DreamerAgent(BaseAgent):
    """An optimistic dreamer who sees possibilities everywhere."""
    
    def __init__(self):
        persona = """You are a dreamer who sees endless possibilities and potential in every topic. 
        You're optimistic, creative, and enthusiastic about new ideas. You think big and 
        aren't afraid to imagine what could be possible. You inspire others with your 
        vision and help them see beyond current limitations. You're the voice of hope 
        and innovation in conversations."""
        
        super().__init__("DreamyDrew", persona)
    
    def respond(self, topic: str) -> str:
        """Generate a dreamy, optimistic response."""
        self.logger.info(f"Generating dreamy response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        return response.content

