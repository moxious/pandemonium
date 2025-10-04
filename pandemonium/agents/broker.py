"""
Broker agent implementation for managing conversation flow.
"""

import logging
from typing import List
from .base_agent import BaseAgent
import random

class BrokerAgent(BaseAgent):
    """A broker agent that manages turn-taking and conversation flow."""
    
    def __init__(self):
        persona = """You are a conversation broker who facilitates discussion between different 
        agents. You introduce topics, manage turn-taking, and ensure everyone gets a chance 
        to speak. You're neutral and objective, focusing on keeping the conversation flowing 
        smoothly while allowing each agent to express their unique perspective.
        
        You try to keep things on topic, and ask people to focus on 1-2 issues rather than
        sprawling a conversation out in a lot of different directions.  You occasionally
        make lists of what you are hearing, requesting one topic be finished before others
        are taken up."""
        
        super().__init__("BrokerBobby", persona)
        self.current_turn = 0
        self.agents = []
    
    def set_agents(self, agents: List[BaseAgent]):
        """Set the list of agents participating in the conversation."""
        self.agents = agents
    
    def introduce_topic(self, topic: str) -> str:
        """Introduce a new topic to the conversation."""
        self.logger.info(f"Introducing topic: {topic}")
        introduction = f"""Topic of the chatroom: {topic}
        Anyone can start"""
        
        self.add_to_history(introduction, "The Broker")
        self.logger.debug(f"Topic introduction added to history: {introduction}")
        return introduction
    
    def get_next_agent(self) -> BaseAgent:
        """Get the next agent in the round-robin rotation."""
        if not self.agents:
            self.logger.error("No agents have been set for the conversation")
            raise ValueError("No agents have been set for the conversation")
        
        agent = self.agents[self.current_turn % len(self.agents)]
        self.current_turn += 1

        if random.random() < 0.1:
            # Broker gets to speak
            return self

        if random.random() < 0.1:
            random_agent = random.choice(self.agents)
            self.logger.debug(f"Random next agent: {random_agent.name} (turn {self.current_turn})")
            return random_agent

        self.logger.debug(f"next agent: {agent.name} (turn {self.current_turn})")
        return agent
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far."""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary_parts = []
        for entry in self.conversation_history:
            summary_parts.append(f"{entry['speaker']}: {entry['message']}")
        
        return "\n".join(summary_parts)
    
    def respond(self, topic: str) -> str:
        """Generate a broker response (typically for transitions)."""
        self.logger.info(f"Generating broker response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        self.logger.debug(f"Broker response generated: {response.content[:100]}...")
        return response.content

