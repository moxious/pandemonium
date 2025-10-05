"""
EvaluatorAgent implementation for conversation evaluation.
"""

import logging
from .base_agent import BaseAgent
from langchain_core.messages import HumanMessage, SystemMessage
from pandemonium.config import Config


class EvaluatorAgent(BaseAgent):
    """A specialized agent for evaluating conversations."""
    
    def __init__(self, evaluation_prompt: str, model=Config.OPENAI_MODEL, temperature=0.3):
        """
        Initialize the EvaluatorAgent with an evaluation prompt.
        
        Args:
            evaluation_prompt: The system prompt containing evaluation instructions
            model: The LLM model to use
            temperature: Temperature for response generation (lower for more focused evaluation)
        """
        super().__init__(
            name="Evaluator", 
            persona=evaluation_prompt,  # Use evaluation prompt as the persona/system message
            model=model, 
            temperature=temperature
        )
        self.logger = logging.getLogger(f"pandemonium.agents.evaluator")
    
    def respond(self, topic: str) -> str:
        """
        Required implementation of abstract method from BaseAgent.
        This method is not used for evaluation - use evaluate_conversation instead.
        """
        raise NotImplementedError("Use evaluate_conversation() method for evaluation")
    
    def evaluate_conversation(self, conversation_history: str) -> str:
        """
        Evaluate a conversation based on the provided history.
        
        Args:
            conversation_history: The full conversation history as a string
            
        Returns:
            The evaluation response
        """
        # Create messages with evaluation prompt as system message and history as user message
        messages = [
            SystemMessage(content=self.persona),
            HumanMessage(content=conversation_history)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
