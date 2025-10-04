"""
Base agent class for all Pandemonium agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pandemonium.config import Config

always_on_prompt = """
Respond briefly, between 1 and 3 sentences. You may use abbreviations and slang,
like you're on an internet chat. Avoid emoji.

In your conversational style, try to balance making your own independent points with 
responding to other users in the chat to keep them engaged. 
You can respond by @Username who said whatever you're responding to.

You may (but are not required to) use tools or search the internet to get information, to fact check, to back up
one of your points, or to find illustrative examples. You share links in the chat if you do.

You may choose a point of view and defend it. Try to make focused points, avoid generating
lists of points or ideas. You are allowed to disagree with other users, please do so 
respectfully. 
"""

class BaseAgent(ABC):
    """Base class for all conversational agents."""
    
    def __init__(self, name: str, persona: str):
        """Initialize the agent with a name and persona description."""
        self.name = name
        self.persona = always_on_prompt + "\n\n" + persona
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=0.7
        )
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_seen_turn = 0  # Track the last turn this agent has seen
        
        # LangGraph memory will be set by the conversation manager
        self.memory_messages: List[BaseMessage] = []
        
        self.logger = logging.getLogger(f"pandemonium.agents.{self.name}")
    
    def add_to_history(self, message: str, speaker: str):
        """Add a message to the agent's conversation history."""
        self.conversation_history.append({
            "speaker": speaker,
            "message": message,
            "turn": len(self.conversation_history) + 1
        })
    
    def get_context(self) -> str:
        """Get the conversation context for this agent."""
        if not self.conversation_history:
            return "This is the beginning of the conversation."
        
        context_parts = []
        for entry in self.conversation_history:
            context_parts.append(f"{entry['speaker']}: {entry['message']}")
        
        return "\n".join(context_parts)
    
    def update_with_new_messages(self, new_messages: List[Dict[str, Any]]):
        """Add new messages to this agent's conversation history and memory."""
        for message in new_messages:
            self.conversation_history.append(message)
            self.last_seen_turn = message.get('turn', 0)
            
            # Add to LangGraph memory for efficient LLM context
            speaker = message.get('speaker', 'Unknown')
            message_content = message.get('message', '')
            
            if speaker != self.name:  # Don't add our own messages to memory
                self.memory_messages.append(HumanMessage(content=f"{speaker}: {message_content}"))
                self.logger.debug(f"Added to memory: {speaker}: {message_content[:50]}...")
    
    def set_memory_messages(self, messages: List[BaseMessage]):
        """Set the memory messages from LangGraph state."""
        self.memory_messages = messages
    
    def get_full_context(self) -> str:
        """Get the complete conversation context for this agent."""
        if not self.conversation_history:
            return "This is the beginning of the conversation."
        
        context_parts = []
        for entry in self.conversation_history:
            context_parts.append(f"{entry['speaker']}: {entry['message']}")
        
        return "\n".join(context_parts)
    
    @abstractmethod
    def respond(self, topic: str) -> str:
        """Generate a response based on the topic and the agent's conversation history."""
        pass
    
    def _add_response_to_memory(self, response: str):
        """Add the agent's own response to memory."""
        self.memory_messages.append(AIMessage(content=response))
        self.logger.debug(f"Added own response to memory: {response[:50]}...")
    
    def _create_messages(self, topic: str) -> List:
        """Create the message list for the LLM using memory for efficient context."""
        # Get recent messages from memory (respecting window size)
        window_size = Config.MEMORY_WINDOW_SIZE
        recent_messages = self.memory_messages[-window_size:] if len(self.memory_messages) > window_size else self.memory_messages
        
        if not recent_messages:
            # First message - include system prompt
            system_prompt = f"""You are {self.name}, a chatroom participant with the following persona:

{self.persona}

You are participating in a round-robin conversation about: {topic}

Respond naturally as your persona would, keeping your response concise but engaging. You can reference, build upon, or disagree with what others have said."""

            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please respond to the conversation about {topic}.")
            ]
        else:
            # Subsequent messages - use memory for context
            system_prompt = f"""You are {self.name}, a chatroom participant with the following persona:

{self.persona}

You are participating in a round-robin conversation about: {topic}

Respond naturally as your persona would, keeping your response concise but engaging. You can reference, build upon, or disagree with what others have said."""

            # Combine system prompt with recent memory messages
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(recent_messages)
            messages.append(HumanMessage(content=f"Please respond to the conversation about {topic}."))
            
            self.logger.debug(f"Using {len(recent_messages)} messages from memory for context")
            return messages

