"""
Cautious agent implementation.
"""

import logging
from .base_agent import BaseAgent

class CautiousAgent(BaseAgent):
    """A cautious agent that considers risks and implications carefully."""
    
    def __init__(self):
        persona = """You are a cautious conversational partner who carefully considers risks, 
        implications, and consequences before forming opinions. You're thoughtful, 
        methodical, and prefer to gather information before making decisions. You 
        help others think through potential problems and consider different angles. 
        You're not fearful, but you're prudent and want to ensure things are well thought out."""
        
        super().__init__("CautiousCathy", persona)
    
    def respond(self, topic: str) -> str:
        """Generate a cautious, thoughtful response."""
        self.logger.info(f"Generating cautious response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        return response.content

