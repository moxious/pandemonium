"""
Conversation management for Pandemonium.
"""

from typing import List, Dict, Any
from .agents import BrokerAgent, MetaAgent, BaseAgent
from .agents.evaluator_agent import EvaluatorAgent
from .langgraph_memory import LangGraphMemory, ConversationState
from langchain_core.messages import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

class Conversation:
    """Manages a conversation between multiple agents."""
    
    def __init__(self, topic: str, agent_specs: List[tuple] = None, evaluation_criteria: str=None):
        """Initialize a conversation with a given topic.
        
        Args:
            topic: The conversation topic
            agent_specs: List of (temperament, expertise) tuples for agent creation.
                        If None, creates 5 random agents.
            evaluation_criteria: The criteria for the broker to evaluate the conversation.
        """
        self.topic = topic
        self.broker = BrokerAgent(topic, evaluation_criteria)
        
        if agent_specs:
            # Create agents based on specifications
            self.agents = []
            for temperament, expertise in agent_specs:
                agent = MetaAgent(temperament=temperament, expertise=expertise)
                self.agents.append(agent)
        else:
            # Default behavior: create 5 random agents
            self.agents = [
                MetaAgent(),
                MetaAgent(),
                MetaAgent(),
                MetaAgent(),
                MetaAgent(),
            ]
        
        self.broker.set_agents(self.agents)
        self.round_count = 0
        self.max_rounds = 3  # Default to 3 rounds
        
        # Initialize LangGraph memory
        self.langgraph_memory = LangGraphMemory()
        self.conversation_state = self.langgraph_memory.create_initial_state(topic, self.max_rounds)
    
    def start_conversation(self) -> str:
        """Start the conversation by introducing the topic."""
        introduction = self.broker.introduce_topic(self.topic)
        
        # Add introduction to LangGraph state
        self.conversation_state = self.langgraph_memory.add_message(
            self.conversation_state, 
            HumanMessage(content=introduction)
        )
        
        return introduction
    
    def next_turn(self) -> str:
        """Get the next agent's response."""
        if self.round_count >= self.max_rounds:
            logger.info(f"Concluding conversation at {self.round_count} rounds")
            return self._conclude_conversation()
        
        agent = self.broker.get_next_agent()
        
        # Get recent messages from LangGraph state for this agent
        recent_messages = self.langgraph_memory.get_recent_messages(self.conversation_state)
        agent.set_memory_messages(recent_messages)
        
        # Get new messages since this agent's last turn (for backward compatibility)
        new_messages = self._get_new_messages_for_agent(agent)
        agent.update_with_new_messages(new_messages)
        
        # Generate response
        logger.info(f"{self.round_count}: Next speaker thinking: {agent.name}")
        response = agent.respond(self.topic)
        
        # Add to broker's history
        self.broker.add_to_history(response, agent.name)
        
        # Add response to LangGraph state
        self.conversation_state = self.langgraph_memory.add_message(
            self.conversation_state,
            AIMessage(content=f"{agent.name}: {response}")
        )
        
        # Check if we've completed a full round
        if self.broker.current_turn % len(self.agents) == 0:
            self.round_count += 1
        
        return f"{agent.name}: {response}"
    
    def _get_new_messages_for_agent(self, agent: BaseAgent) -> List[Dict[str, Any]]:
        """Get only the messages that this agent hasn't seen yet."""
        # Get all conversation history from broker
        all_messages = self.broker.conversation_history
        
        # Filter to only messages after this agent's last seen turn
        new_messages = []
        for message in all_messages:
            message_turn = message.get('turn', 0)
            if message_turn > agent.last_seen_turn:
                new_messages.append(message)
        
        return new_messages
    
    def _format_conversation_history(self) -> str:
        """Format the conversation history as a readable string for evaluation."""
        # Get all messages from the broker's conversation history
        all_messages = self.broker.conversation_history
        
        if not all_messages:
            return "No conversation history available."
        
        # Format each message as "Speaker: Message"
        formatted_messages = []
        for message in all_messages:
            speaker = message.get('speaker', 'Unknown')
            content = message.get('message', '')
            formatted_messages.append(f"{speaker}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _conclude_conversation(self) -> str:
        """Conclude the conversation with a fresh evaluator agent."""
        logger.info(f"Concluding / summarizing conversation")
        
        # Create comprehensive evaluation prompt
        evaluation_prompt = f"""
You are an independent evaluator reviewing a complete chatroom discussion about "{self.topic}".

You have access to the entire conversation history. 

First, you will read your evaluation criteria, these frame how you understand the
conversation.  Your evaluation criteria are: {self.broker.evaluation_criteria}

Next, you will read through the entire chatroom discussion.

Next, you will synthesize the best outcome of the conversation, based on the evaluation criteria.

Finally, you will provide a brief summary of the best parts of the conversation, and end with a 
statement that says:

Result: (your synthesis of the best outcome in no more than 3 sentences)
"""
        
        evaluator = EvaluatorAgent(evaluation_prompt, model='gpt-5', temperature=0.5)
        conversation_history = self._format_conversation_history()
        evaluator_summary = evaluator.evaluate_conversation(conversation_history)
        
        conclusion = f"""\n--- Conversation Complete ---

We've completed {self.max_rounds} rounds of discussion on "{self.topic}". 
Thank you to all participants for sharing their unique perspectives!

Final evaluation by independent assessor:
{evaluator.name}: {evaluator_summary}"""
        
        # Add conclusion to LangGraph state
        self.conversation_state = self.langgraph_memory.add_message(
            self.conversation_state,
            HumanMessage(content=conclusion)
        )
        
        return conclusion
    
    def set_max_rounds(self, rounds: int):
        """Set the maximum number of conversation rounds."""
        self.max_rounds = rounds
        self.conversation_state["max_rounds"] = rounds
    
    def get_conversation_state(self) -> ConversationState:
        """Get the current LangGraph conversation state."""
        return self.conversation_state
    
    def set_conversation_state(self, state: ConversationState):
        """Set the LangGraph conversation state (for loading saved conversations)."""
        self.conversation_state = state
        self.round_count = state.get("round_count", 0)
        self.topic = state.get("topic", self.topic)
