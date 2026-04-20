"""
Pluggable turn-taking strategies for Pandemonium conversations.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict


class TurnStrategy(ABC):
    """Abstract base class for conversation turn-taking strategies."""

    @abstractmethod
    def plan_round(self, agent_names: List[str], broker_name: str, round_num: int) -> List[str]:
        """Return ordered list of speaker names for the upcoming round.

        Args:
            agent_names: Names of the participating agents (not including broker).
            broker_name: Name of the broker agent.
            round_num: The round number (0-indexed).

        Returns:
            Ordered list of speaker names for this round.
        """
        ...

    def is_round_complete(self, speakers_this_round: List[str], planned_order: List[str]) -> bool:
        """Check if all planned speakers have spoken.

        Args:
            speakers_this_round: Names of agents who have spoken so far this round.
            planned_order: The planned speaker order for this round.

        Returns:
            True if the round is complete.
        """
        return len(speakers_this_round) >= len(planned_order)

    @property
    def is_message_based(self) -> bool:
        """If True, the graph uses per-message speaker selection instead of pre-planned rounds."""
        return False

    def select_next_speaker(
        self,
        agent_names: List[str],
        broker_name: str,
        messages: List,
        speaker_history: List[str],
        agent_metadata: Dict[str, Dict[str, str]],
        message_count: int,
        max_messages: int,
    ) -> str:
        """Select the next speaker based on conversation dynamics.

        Only called when is_message_based is True.
        """
        raise NotImplementedError("Message-based strategies must implement select_next_speaker()")


class RoundRobinStrategy(TurnStrategy):
    """Fixed order, every agent speaks once per round. Broker never interjects."""

    def plan_round(self, agent_names: List[str], broker_name: str, round_num: int) -> List[str]:
        return list(agent_names)


class BrokerMediatedStrategy(TurnStrategy):
    """All agents speak once per round. Broker interjects after every N agent turns.

    Broker turns don't count toward round completion — the round ends when
    all agents have spoken.
    """

    def __init__(self, broker_every_n: int = 3):
        self.broker_every_n = broker_every_n

    def plan_round(self, agent_names: List[str], broker_name: str, round_num: int) -> List[str]:
        order = []
        agent_count = 0
        for name in agent_names:
            order.append(name)
            agent_count += 1
            if agent_count % self.broker_every_n == 0 and agent_count < len(agent_names):
                order.append(broker_name)
        return order


class StochasticStrategy(TurnStrategy):
    """Shuffled order per round. Broker has a configurable probability of interjecting.

    Guarantees every agent speaks at least once per round.
    """

    def __init__(self, broker_interjection_prob: float = 0.3):
        self.broker_interjection_prob = broker_interjection_prob

    def plan_round(self, agent_names: List[str], broker_name: str, round_num: int) -> List[str]:
        shuffled = list(agent_names)
        random.shuffle(shuffled)

        order = []
        for name in shuffled:
            if order and random.random() < self.broker_interjection_prob:
                order.append(broker_name)
            order.append(name)
        return order


STRATEGIES = {
    "round_robin": RoundRobinStrategy,
    "broker_mediated": BrokerMediatedStrategy,
    "stochastic": StochasticStrategy,
}


def get_strategy(name: str, **kwargs) -> TurnStrategy:
    """Get a turn strategy by name.

    Args:
        name: Strategy name (round_robin, broker_mediated, stochastic, chatroom).
        **kwargs: Strategy-specific configuration.

    Returns:
        An instantiated TurnStrategy.
    """
    if name == "chatroom":
        from pandemonium.chatroom_strategy import ChatRoomStrategy
        return ChatRoomStrategy(**kwargs)
    if name not in STRATEGIES:
        available = list(STRATEGIES.keys()) + ["chatroom"]
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGIES[name](**kwargs)
