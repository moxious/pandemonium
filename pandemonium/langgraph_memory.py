"""
LangGraph memory implementation for Pandemonium agents.
"""

from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pandemonium.config import Config


class ConversationState(TypedDict):
    """State schema for LangGraph conversation management."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    current_agent: str
    topic: str
    round_count: int
    max_rounds: int
    agent_responses: Annotated[Dict[str, str], "Agent responses in current round"]


class LangGraphMemory:
    """LangGraph-based memory management for Pandemonium agents."""
    
    def __init__(self, window_size: int = None):
        """Initialize LangGraph memory with optional window size."""
        self.window_size = window_size or Config.MEMORY_WINDOW_SIZE
        self.graph = self._create_conversation_graph()
        self.checkpointer = None  # Will be set when persistence is needed
    
    def _create_conversation_graph(self) -> StateGraph:
        """Create the LangGraph conversation management graph."""
        graph = StateGraph(ConversationState)
        
        # Add nodes for different conversation phases
        graph.add_node("introduce_topic", self._introduce_topic)
        graph.add_node("get_agent_response", self._get_agent_response)
        graph.add_node("check_round_complete", self._check_round_complete)
        graph.add_node("conclude_conversation", self._conclude_conversation)
        
        # Define the conversation flow
        graph.set_entry_point("introduce_topic")
        graph.add_edge("introduce_topic", "get_agent_response")
        graph.add_edge("get_agent_response", "check_round_complete")
        
        # Conditional edges based on conversation state
        graph.add_conditional_edges(
            "check_round_complete",
            self._should_continue,
            {
                "continue": "get_agent_response",
                "conclude": "conclude_conversation",
                "end": END
            }
        )
        
        graph.add_edge("conclude_conversation", END)
        
        return graph.compile()
    
    def _introduce_topic(self, state: ConversationState) -> ConversationState:
        """Introduce the conversation topic."""
        topic = state["topic"]
        introduction = f"Welcome to our discussion about: {topic}\n\nLet's begin our conversation!"
        
        state["messages"].append(SystemMessage(content=introduction))
        state["round_count"] = 0
        state["agent_responses"] = {}
        
        return state
    
    def _get_agent_response(self, state: ConversationState) -> ConversationState:
        """Get response from the current agent."""
        # This will be implemented by the specific agent
        # For now, we'll add a placeholder
        agent_name = state["current_agent"]
        response = f"{agent_name} is thinking about the topic..."
        
        state["messages"].append(AIMessage(content=response))
        state["agent_responses"][agent_name] = response
        
        return state
    
    def _check_round_complete(self, state: ConversationState) -> ConversationState:
        """Check if the current round is complete."""
        # This logic will be implemented based on the number of agents
        # For now, we'll increment the round count
        state["round_count"] += 1
        return state
    
    def _conclude_conversation(self, state: ConversationState) -> ConversationState:
        """Conclude the conversation."""
        conclusion = f"\n--- Conversation Complete ---\n\nWe've completed {state['round_count']} rounds of discussion on \"{state['topic']}\".\nThank you to all participants for sharing their unique perspectives!"
        
        state["messages"].append(SystemMessage(content=conclusion))
        return state
    
    def _should_continue(self, state: ConversationState) -> str:
        """Determine if the conversation should continue."""
        if state["round_count"] >= state["max_rounds"]:
            return "conclude"
        return "continue"
    
    def get_recent_messages(self, state: ConversationState, window_size: int = None) -> List[BaseMessage]:
        """Get recent messages within the specified window size."""
        window = window_size or self.window_size
        messages = state["messages"]
        
        if len(messages) <= window:
            return messages
        
        # Return the last 'window' messages
        return messages[-window:]
    
    def add_message(self, state: ConversationState, message: BaseMessage) -> ConversationState:
        """Add a message to the conversation state."""
        state["messages"].append(message)
        return state
    
    def get_conversation_history(self, state: ConversationState) -> List[Dict[str, Any]]:
        """Convert LangGraph messages to the format expected by existing code."""
        history = []
        for i, message in enumerate(state["messages"]):
            if isinstance(message, HumanMessage):
                speaker = "Human"
            elif isinstance(message, AIMessage):
                speaker = "AI"
            elif isinstance(message, SystemMessage):
                speaker = "System"
            else:
                speaker = "Unknown"
            
            history.append({
                "speaker": speaker,
                "message": message.content,
                "turn": i + 1
            })
        
        return history
    
    def set_persistence(self, checkpointer):
        """Set up persistence for the graph."""
        self.checkpointer = checkpointer
        self.graph = self.graph.compile(checkpointer=checkpointer)
    
    def create_initial_state(self, topic: str, max_rounds: int = 3) -> ConversationState:
        """Create initial conversation state."""
        return ConversationState(
            messages=[],
            current_agent="",
            topic=topic,
            round_count=0,
            max_rounds=max_rounds,
            agent_responses={}
        )
