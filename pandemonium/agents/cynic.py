"""
Cynic agent implementation.
"""

import logging
from .base_agent import BaseAgent


class CynicAgent(BaseAgent):
    """A cynical conversational agent that questions everything."""
    
    def __init__(self):
        persona = """You are a cynic who approaches every topic with skepticism and doubt. 
        You question assumptions, point out potential problems, and often see the negative 
        side of things. You're not necessarily pessimistic, but you're realistic about 
        challenges and limitations. You challenge others' ideas constructively and 
        encourage critical thinking."""
        
        super().__init__("CynicalCedric", persona)
    
    def respond(self, topic: str) -> str:
        """Generate a cynical response."""
        self.logger.info(f"Generating cynical response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        return response.content

