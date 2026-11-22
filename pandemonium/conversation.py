"""
Conversation factory/facade for Pandemonium.

Constructs agents, configures the LangGraph conversation graph,
and provides a simple interface for running conversations.
"""

from typing import List, Iterator, Dict, Any, Optional
from .agents import BrokerAgent, MetaAgent
from .config import TokenTracker
from .graph import build_conversation_graph, ConversationState
from .turn_strategies import TurnStrategy, get_strategy
import logging

logger = logging.getLogger(__name__)

# Message cap for message-based strategies when the caller sets none.
DEFAULT_MAX_MESSAGES = 30


class Conversation:
    """Factory that builds agents and runs conversations via LangGraph."""

    def __init__(
        self,
        topic: str,
        agent_specs: List = None,
        evaluation_criteria: str = None,
        strategy_name: str = "broker_mediated",
        strategy_kwargs: dict = None,
        transcript_logger=None,
        broker_config: dict = None,
        broker_mode: str = "silent",
    ):
        """Initialize a conversation.

        Args:
            topic: The conversation topic.
            agent_specs: List of (temperament, expertise, trait) tuples or dicts
                        with keys: temperament, expertise, trait, provider, model.
                        If None, creates 5 random agents.
            evaluation_criteria: Criteria for the evaluator.
            strategy_name: Turn strategy name (round_robin, broker_mediated, stochastic).
            strategy_kwargs: Extra kwargs for the turn strategy constructor.
            transcript_logger: Optional TranscriptLogger instance.
            broker_config: Optional dict with provider/model for the broker agent.
        """
        self.topic = topic
        self.evaluation_criteria = evaluation_criteria or "pick most interesting or important issues"

        # Build broker
        broker_kwargs = {}
        if broker_config:
            if broker_config.get("provider"):
                broker_kwargs["provider"] = broker_config["provider"]
            if broker_config.get("model"):
                broker_kwargs["model"] = broker_config["model"]
        self.broker = BrokerAgent(topic, self.evaluation_criteria, **broker_kwargs)

        # Build agents
        if agent_specs:
            self.agents = []
            for spec in agent_specs:
                if isinstance(spec, dict):
                    temperament = spec.get("temperament")
                    expertise = spec.get("expertise")
                    trait = spec.get("trait", "generalist")
                    model = spec.get("model")
                    provider = spec.get("provider")
                else:
                    temperament, expertise, trait = spec[0], spec[1], spec[2]
                    model = None
                    provider = None
                logger.info(f"Creating agent temperament={temperament}, expertise={expertise}, trait={trait}, provider={provider}, model={model}")
                agent = MetaAgent(temperament=temperament, expertise=expertise, trait=trait,
                                  model=model, provider=provider,
                                  taken_names=self._taken_names())
                self.agents.append(agent)
        else:
            self.agents = []
            for _ in range(5):
                self.agents.append(MetaAgent(taken_names=self._taken_names()))

        # Broker needs agent list for introduction
        self.broker.set_agents(self.agents)

        # Build turn strategy
        self.turn_strategy = get_strategy(strategy_name, **(strategy_kwargs or {}))

        # Build agent registry (name -> agent)
        self.agent_registry = {agent.name: agent for agent in self.agents}

        # Transcript logger (may be None)
        self.transcript_logger = transcript_logger

        # Broker mode: "silent" or "active"
        self.broker_mode = broker_mode

        # Compile graph
        self.graph = build_conversation_graph()

        # Default rounds / messages
        self.max_rounds = 3
        self.max_messages = 0  # 0 means round-based mode

    def _taken_names(self) -> set:
        """Names already in the room, so new agents never collide with them."""
        return {self.broker.name} | {agent.name for agent in self.agents}

    def set_max_rounds(self, rounds: int):
        """Set the maximum number of conversation rounds."""
        self.max_rounds = rounds

    def set_max_messages(self, messages: int):
        """Set the maximum number of messages for message-based strategies."""
        self.max_messages = messages

    def run(self) -> Iterator[Dict[str, Any]]:
        """Run the conversation, yielding events as they occur.

        Yields dicts with keys:
            type: "message" | "complete"
            speaker: speaker name
            content: message content
        """
        max_messages = self.max_messages
        if self.turn_strategy.is_message_based and max_messages <= 0:
            # Message-based strategies never advance rounds, so they need a
            # message cap or the graph never reaches evaluation.
            max_messages = DEFAULT_MAX_MESSAGES

        initial_state: ConversationState = {
            "messages": [],
            "topic": self.topic,
            "round_count": 0,
            "max_rounds": self.max_rounds,
            "current_speaker": "",
            "speakers_this_round": [],
            "planned_round_order": [],
            "evaluation_criteria": self.evaluation_criteria,
            "round_boundaries": [],
            "summary": "",
            "message_count": 0,
            "max_messages": max_messages,
            "speaker_history": [],
            "consecutive_failures": 0,
        }

        # should_continue() is what ends the run; the recursion limit is only a
        # backstop against a runaway loop, so size it for the worst case.
        # Each turn visits 3 nodes (select → respond → check).
        if max_messages > 0:
            max_turns = max_messages
        else:
            # A round can plan at most one broker turn between each pair of
            # agents (stochastic with p=1.0), i.e. 2n - 1 turns.
            max_turns = self.max_rounds * (2 * len(self.agent_registry))
        step_limit = 2 + (max_turns * 3) + 20

        self.token_tracker = TokenTracker()
        config = {
            "recursion_limit": max(50, step_limit),
            "configurable": {
                "agents": self.agent_registry,
                "broker": self.broker,
                "turn_strategy": self.turn_strategy,
                "transcript_logger": self.transcript_logger,
                "token_tracker": self.token_tracker,
                "broker_mode": self.broker_mode,
            },
        }

        for event in self.graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, state_update in event.items():
                new_messages = state_update.get("messages", [])
                for msg in new_messages:
                    content = msg.content
                    if not content.strip():
                        continue

                    # Classify by the node that produced the message, never by
                    # content: an agent may say anything.
                    if node_name == "evaluate":
                        yield {"type": "complete", "speaker": "system", "content": content}
                    elif node_name == "introduce_topic":
                        yield {"type": "message", "speaker": "Broker", "content": content}
                    elif node_name == "agent_respond":
                        yield {"type": "message", "speaker": state_update["current_speaker"], "content": content}
