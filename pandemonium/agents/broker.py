"""
Broker agent implementation for managing conversation flow.
"""

from typing import List
from .base_agent import BaseAgent

class BrokerAgent(BaseAgent):
    """A broker agent that manages turn-taking and conversation flow."""
    
    def __init__(self, topic: str, evaluation_criteria: str = "pick most interesting or important issues",
                 model='gpt-5-mini-2025-08-07', provider='openai'):
        self.topic = topic
        self.evaluation_criteria = evaluation_criteria

        persona = f"""You are a conversation broker who facilitates discussion between different
        chatroom participants. You summarize, manage turn-taking, and ensure everyone gets a chance
        to speak. You're neutral and objective, focusing on keeping the conversation flowing.

        The topic of the conversation will be known to all participants: "{self.topic}"

        Chat room participants can get wild, so one of your most important jobs is to summarize,
        and focus conversation, according to your evaluation criteria. You do not
        summarize EVERYTHING, you choose & focus on the most important issues to prune discussion.

        Evaluation criteria are there to ensure the conversation produces a coherent result.

        Your evaluation criteria are: {self.evaluation_criteria}

        You understand that focusing conversation harnesses the group's collective intelligence.
        """

        super().__init__("BrokerBobby", persona, model=model, provider=provider)
        self.agents = []
    
    def set_agents(self, agents: List[BaseAgent]):
        """Set the list of agents participating in the conversation."""
        self.agents = agents
    
    def introduce_topic(self, topic: str) -> str:
        """Introduce a new topic to the conversation."""
        self.logger.info(f"Introducing topic: {topic}")
        introduction = f"""Topic of the chatroom: {topic}

        The following users are participating in the conversation:
        {", ".join([agent.name for agent in self.agents])}"""

        self.logger.debug(f"Topic introduction added to history: {introduction}")
        return introduction
    
