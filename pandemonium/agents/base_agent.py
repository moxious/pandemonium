"""
Base agent class for all Pandemonium agents.
"""

import logging
from abc import ABC
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pandemonium.config import Config, create_chat_model

class BaseAgent(ABC):
    """Base class for all conversational agents."""

    def __init__(self, name: str, persona: str, model=None, provider=None, temperature=None):
        """Initialize the agent with a name and persona description."""
        self.name = name
        self.persona = persona

        self.llm = create_chat_model(provider=provider, model=model, temperature=temperature)

        self.logger = logging.getLogger(f"pandemonium.agents.{self.name}")

    def respond(self, topic: str, context_messages: List[BaseMessage] = None) -> dict:
        """Generate a response based on the topic and conversation context.

        Returns dict with 'content', 'input_tokens', 'output_tokens'.
        """
        messages = self._create_messages(topic, context_messages)
        response = self.llm.invoke(messages)
        usage = getattr(response, 'usage_metadata', None) or {}
        return {
            "content": response.content,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    def _create_messages(self, topic: str, context_messages: List[BaseMessage] = None) -> List:
        """Create the message list for the LLM.

        Context windowing is handled by the graph before calling respond(),
        so this method trusts whatever messages it receives.
        """
        if not context_messages:
            return [
                SystemMessage(content=self.persona),
                HumanMessage(content=f"Please respond to the conversation about {topic}.")
            ]

        messages = [SystemMessage(content=self.persona)]
        messages.extend(context_messages)
        messages.append(HumanMessage(content=f"Playing your persona, please contribute to the discussion."))

        self.logger.debug(f"Using {len(context_messages)} messages for context")
        return messages

