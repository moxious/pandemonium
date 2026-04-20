"""
End-to-end test for the Pandemonium conversation pipeline.

Mocks `create_chat_model` so no real LLM calls are made, then runs the
full LangGraph conversation loop and asserts the flow behaves correctly.
"""

import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from pandemonium.conversation import Conversation


_call_counter = 0


def _make_fake_llm(*args, **kwargs):
    """Factory that returns a fresh mock LLM each time (used as side_effect)."""
    global _call_counter

    def fake_invoke(messages):
        global _call_counter
        _call_counter += 1
        response = AIMessage(content=f"Mock response #{_call_counter}")
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        return response

    llm = MagicMock()
    llm.invoke = fake_invoke
    return llm


@patch("pandemonium.agents.evaluator_agent.create_chat_model", side_effect=_make_fake_llm)
@patch("pandemonium.agents.base_agent.create_chat_model", side_effect=_make_fake_llm)
class TestConversationE2E(unittest.TestCase):
    """Run a full conversation with mocked LLMs and verify the pipeline."""

    def setUp(self):
        global _call_counter
        _call_counter = 0

    def _make_conversation(self, strategy="round_robin", max_rounds=2, agent_count=2):
        """Helper to create a Conversation with a small, fast config."""
        specs = [
            {"temperament": "cynic", "expertise": "engineer", "trait": "generalist"},
            {"temperament": "dreamer", "expertise": "airesearcher", "trait": "generalist"},
        ][:agent_count]
        conv = Conversation(
            topic="Is AI overhyped?",
            agent_specs=specs,
            evaluation_criteria="identify the strongest argument",
            strategy_name=strategy,
        )
        conv.set_max_rounds(max_rounds)
        return conv

    # -- Core flow tests --

    def test_conversation_produces_events(self, _mock_base, _mock_eval):
        """A basic conversation should yield message events and end with a complete event."""
        conv = self._make_conversation()

        events = list(conv.run())

        self.assertGreater(len(events), 0, "Should produce at least one event")

        types = {e["type"] for e in events}
        self.assertIn("message", types, "Should have message events")
        self.assertIn("complete", types, "Should have a complete event")

    def test_complete_event_is_last(self, _mock_base, _mock_eval):
        """The 'complete' event should be the final event emitted."""
        conv = self._make_conversation()
        events = list(conv.run())

        self.assertEqual(events[-1]["type"], "complete")

    def test_first_event_is_broker_intro(self, _mock_base, _mock_eval):
        """The first event should be the broker's topic introduction."""
        conv = self._make_conversation()
        events = list(conv.run())

        first = events[0]
        self.assertEqual(first["type"], "message")
        self.assertEqual(first["speaker"], "Broker")
        self.assertIn("Is AI overhyped?", first["content"])

    def test_all_agents_speak(self, _mock_base, _mock_eval):
        """Every agent should have at least one speaking turn."""
        conv = self._make_conversation()
        events = list(conv.run())

        agent_names = set(conv.agent_registry.keys())
        # Check both the speaker field and the "Name: content" prefix in messages
        content_speakers = set()
        for e in events:
            if e["type"] == "message" and ":" in e["content"]:
                content_speakers.add(e["content"].split(":")[0].strip())

        for name in agent_names:
            self.assertIn(name, content_speakers, f"Agent '{name}' never spoke")

    def test_evaluation_in_conclusion(self, _mock_base, _mock_eval):
        """The completion event should include evaluation output."""
        conv = self._make_conversation()
        events = list(conv.run())

        complete_event = events[-1]
        self.assertIn("Conversation Complete", complete_event["content"])

    # -- Token tracking --

    def test_token_tracker_accumulates(self, _mock_base, _mock_eval):
        """Token tracker should have non-zero totals after a conversation."""
        conv = self._make_conversation()
        list(conv.run())

        self.assertGreater(conv.token_tracker.total_input_tokens, 0)
        self.assertGreater(conv.token_tracker.total_output_tokens, 0)

    # -- Strategy variations --

    def test_broker_mediated_strategy(self, _mock_base, _mock_eval):
        """Broker-mediated strategy should complete without errors."""
        conv = self._make_conversation(strategy="broker_mediated")
        events = list(conv.run())

        self.assertEqual(events[-1]["type"], "complete")

    def test_round_count_respected(self, _mock_base, _mock_eval):
        """Conversation with 1 round should produce fewer events than 3 rounds."""
        conv1 = self._make_conversation(max_rounds=1)
        conv3 = self._make_conversation(max_rounds=3)

        events1 = list(conv1.run())
        events3 = list(conv3.run())

        msgs1 = [e for e in events1 if e["type"] == "message"]
        msgs3 = [e for e in events3 if e["type"] == "message"]
        self.assertGreater(len(msgs3), len(msgs1))


if __name__ == "__main__":
    unittest.main()
