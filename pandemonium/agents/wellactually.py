"""
WellActually agent implementation.
"""

import logging
from .base_agent import BaseAgent

class WellActuallyAgent(BaseAgent):
    """A fact-checking agent."""
    
    def __init__(self):
        persona = """You are a stickler for facts and like to research things before speaking.
        You are respectful but you hold others to factual statements and will let them know when
        the things they say aren't really true. You may give opinions about the topic, but only if
        you reasonably know the opinion not to contradict the facts.
        """

        super().__init__("WellActuallyWally", persona)
    
    def respond(self, topic: str) -> str:
        """Generate a fact-checking response."""
        self.logger.info(f"Generating fact-checking response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        return response.content

