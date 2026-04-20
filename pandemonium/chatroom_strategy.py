"""
ChatRoom turn-taking strategy for Pandemonium.

Selects speakers per-message using urgency scores rather than
pre-planning entire rounds. Produces organic chatroom dynamics:
dominant voices, lurkers, rapid exchanges, and contextual broker
interjections.
"""

import random
from typing import List, Dict

from langchain_core.messages import BaseMessage

from pandemonium.turn_strategies import TurnStrategy


# Temperament -> base speaking impulse (higher = more eager to speak)
TEMPERAMENT_IMPULSE = {
    "chaos": 0.9,
    "edgy": 0.85,
    "scamp": 0.8,
    "creative": 0.7,
    "dreamer": 0.7,
    "power": 0.7,
    "wellactually": 0.65,
    "detective": 0.6,
    "optimist": 0.6,
    "focused": 0.6,
    "nononsense": 0.55,
    "theory": 0.55,
    "pragmatic": 0.5,
    "sunny": 0.5,
    "empath": 0.5,
    "dad": 0.45,
    "mom": 0.45,
    "helper": 0.45,
    "cautious": 0.3,
    "bland": 0.25,
}
DEFAULT_IMPULSE = 0.5

BROKER_BASE_URGENCY = 0.3
BROKER_URGENCY_RATE = 0.15
BROKER_INTERVENTION_THRESHOLD = 5


class ChatRoomStrategy(TurnStrategy):
    """Message-based turn strategy using urgency scores.

    No rounds are planned upfront. After each message, every agent is
    scored and the highest-scoring agent speaks next.
    """

    def __init__(self, noise_scale: float = 0.15, consecutive_penalty: float = 0.8):
        self.noise_scale = noise_scale
        self.consecutive_penalty = consecutive_penalty

    @property
    def is_message_based(self) -> bool:
        return True

    # --- ABC stubs (unused in message-based mode) ---

    def plan_round(self, agent_names: List[str], broker_name: str, round_num: int) -> List[str]:
        return []

    def is_round_complete(self, speakers_this_round: List[str], planned_order: List[str]) -> bool:
        return True

    # --- Per-message speaker selection ---

    def select_next_speaker(
        self,
        agent_names: List[str],
        broker_name: str,
        messages: List[BaseMessage],
        speaker_history: List[str],
        agent_metadata: Dict[str, Dict[str, str]],
        message_count: int,
        max_messages: int,
    ) -> str:
        scores: Dict[str, float] = {}

        for name in agent_names:
            meta = agent_metadata.get(name, {})
            temperament = meta.get("temperament", "")

            # 1. Base impulse from temperament
            base = TEMPERAMENT_IMPULSE.get(temperament, DEFAULT_IMPULSE)

            # 2. Recency pressure: grows with silence, capped
            recency = self._messages_since_last_spoke(name, speaker_history)
            recency_score = min(recency * 0.12, 0.6)

            # 3. Mention boost: was agent named or their expertise referenced?
            mention = 0.0
            if messages:
                last_content = messages[-1].content.lower()
                if name.lower() in last_content:
                    mention = 0.35
                elif meta.get("expertise", "").lower() in last_content:
                    mention = 0.15

            # 4. Consecutive penalty: dampening if agent just spoke
            consecutive = 0.0
            if speaker_history and speaker_history[-1] == name:
                consecutive = -self.consecutive_penalty

            # 5. Random noise for non-determinism
            noise = random.gauss(0, self.noise_scale)

            scores[name] = base + recency_score + mention + consecutive + noise

        # Broker urgency: separate track (skipped when broker_name is None / silent mode)
        if broker_name is not None:
            broker_recency = self._messages_since_last_spoke(broker_name, speaker_history)
            broker_score = BROKER_BASE_URGENCY + (broker_recency * BROKER_URGENCY_RATE)
            if broker_recency >= BROKER_INTERVENTION_THRESHOLD:
                broker_score += 0.4
            if speaker_history and speaker_history[-1] == broker_name:
                broker_score -= self.consecutive_penalty
            broker_score += random.gauss(0, self.noise_scale)
            scores[broker_name] = broker_score

        return max(scores, key=scores.get)

    @staticmethod
    def _messages_since_last_spoke(name: str, speaker_history: List[str]) -> int:
        """Count messages since `name` last spoke. Returns len(history) if never spoke."""
        for i, speaker in enumerate(reversed(speaker_history)):
            if speaker == name:
                return i
        return len(speaker_history)
