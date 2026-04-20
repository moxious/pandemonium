"""
EvaluatorAgent implementation for conversation evaluation.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from pandemonium.config import create_chat_model


class EvaluatorAgent:
    """A standalone agent for evaluating conversations. Not a conversational participant."""

    def __init__(self, evaluation_prompt: str, model=None, provider=None, temperature=0.3):
        self.name = "Evaluator"
        self.persona = evaluation_prompt
        self.llm = create_chat_model(provider=provider, model=model, temperature=temperature)
        self.logger = logging.getLogger("pandemonium.agents.evaluator")

    def evaluate_conversation(self, conversation_history: str) -> dict:
        """Evaluate a conversation based on the provided history.

        Returns dict with 'content', 'input_tokens', 'output_tokens'.
        """
        messages = [
            SystemMessage(content=self.persona),
            HumanMessage(content=conversation_history)
        ]
        response = self.llm.invoke(messages)
        usage = getattr(response, 'usage_metadata', None) or {}
        return {
            "content": response.content,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
