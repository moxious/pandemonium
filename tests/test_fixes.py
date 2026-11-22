"""
Regression tests for orchestration bugs found in the design review.

Like test_e2e, these patch `create_chat_model` at both import sites so no
API keys or network access are needed.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from pandemonium.chatroom_strategy import _mentions
from pandemonium.conversation import Conversation, DEFAULT_MAX_MESSAGES
from pandemonium.transcript import TranscriptLogger


def _fake_llm_factory(content="Mock response", fail_when=None):
    """Return a create_chat_model side_effect whose LLMs reply with `content`.

    fail_when(call_number) -> True makes that call raise.
    """
    calls = {"n": 0}

    def factory(*args, **kwargs):
        def invoke(messages):
            calls["n"] += 1
            if fail_when and fail_when(calls["n"]):
                raise RuntimeError("simulated provider error")
            response = AIMessage(content=content)
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
            return response

        llm = MagicMock()
        llm.invoke = invoke
        return llm

    return factory


SPECS = [
    {"temperament": "cynic", "expertise": "engineer", "trait": "generalist"},
    {"temperament": "dreamer", "expertise": "airesearcher", "trait": "focused"},
    {"temperament": "optimist", "expertise": "security", "trait": "chaos"},
]


class _PatchedTestCase(unittest.TestCase):
    """Patch both create_chat_model import sites with a configurable fake."""

    llm_content = "Mock response"
    fail_when = None

    def setUp(self):
        factory = _fake_llm_factory(self.llm_content, self.fail_when)
        self.base_patch = patch("pandemonium.agents.base_agent.create_chat_model", side_effect=factory)
        self.eval_patch = patch("pandemonium.agents.evaluator_agent.create_chat_model", side_effect=factory)
        self.mock_base = self.base_patch.start()
        self.mock_eval = self.eval_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.addCleanup(self.eval_patch.stop)


class TestEventContract(_PatchedTestCase):

    def test_agent_message_events_carry_speaker(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:2], strategy_name="round_robin")
        conv.set_max_rounds(1)
        events = list(conv.run())

        agent_events = [e for e in events if e["type"] == "message" and e["speaker"] != "Broker"]
        self.assertEqual(len(agent_events), 2)
        for e in agent_events:
            self.assertIn(e["speaker"], conv.agent_registry)
            self.assertTrue(e["content"].startswith(f"{e['speaker']}: "))


class TestCompleteDetection(_PatchedTestCase):
    llm_content = "--- Conversation Complete --- just kidding"

    def test_only_evaluator_output_is_complete(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:2], strategy_name="round_robin")
        conv.set_max_rounds(1)
        events = list(conv.run())

        complete = [i for i, e in enumerate(events) if e["type"] == "complete"]
        self.assertEqual(complete, [len(events) - 1])


class TestAgentNames(_PatchedTestCase):

    def test_identical_specs_get_unique_names(self):
        specs = [dict(SPECS[0]) for _ in range(12)]  # more than the 10 random suffixes
        conv = Conversation(topic="t", agent_specs=specs, strategy_name="round_robin")
        self.assertEqual(len(conv.agent_registry), 12)

    def test_random_cast_has_five_agents(self):
        conv = Conversation(topic="t")
        self.assertEqual(len(conv.agent_registry), 5)


class TestProviderDefaults(_PatchedTestCase):

    def test_broker_and_evaluator_use_configured_defaults(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:1], strategy_name="round_robin")
        conv.set_max_rounds(1)
        list(conv.run())

        for mock in (self.mock_base, self.mock_eval):
            for call in mock.call_args_list:
                self.assertIsNone(call.kwargs.get("provider"))
                self.assertIsNone(call.kwargs.get("model"))

    def test_broker_config_still_overrides(self):
        Conversation(topic="t", agent_specs=SPECS[:1],
                     broker_config={"provider": "anthropic", "model": "some-model"})
        broker_call = self.mock_base.call_args_list[0]
        self.assertEqual(broker_call.kwargs["provider"], "anthropic")
        self.assertEqual(broker_call.kwargs["model"], "some-model")


class TestRunLength(_PatchedTestCase):

    def test_stochastic_active_worst_case_completes(self):
        # p=1.0 plans a broker turn between every pair of agents: 2n-1 turns
        # per round, which overran the old recursion limit estimate.
        conv = Conversation(topic="t", agent_specs=SPECS, strategy_name="stochastic",
                            strategy_kwargs={"broker_interjection_prob": 1.0},
                            broker_mode="active")
        conv.set_max_rounds(5)
        events = list(conv.run())

        self.assertEqual(events[-1]["type"], "complete")
        turns = [e for e in events if e["type"] == "message" and e["speaker"] != "Broker"]
        self.assertEqual(len(turns), 5 * (2 * len(SPECS) - 1))

    def test_chatroom_without_message_cap_terminates(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:2], strategy_name="chatroom")
        events = list(conv.run())

        self.assertEqual(events[-1]["type"], "complete")
        turns = [e for e in events if e["type"] == "message" and e["speaker"] != "Broker"]
        self.assertEqual(len(turns), DEFAULT_MAX_MESSAGES)


class TestTurnFailureSkipped(_PatchedTestCase):
    # Call 1 is the first agent turn (the intro makes no LLM call).
    fail_when = staticmethod(lambda n: n == 1)

    def test_single_failed_turn_is_skipped(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:2], strategy_name="round_robin")
        conv.set_max_rounds(2)
        events = list(conv.run())

        self.assertEqual(events[-1]["type"], "complete")
        turns = [e for e in events if e["type"] == "message" and e["speaker"] != "Broker"]
        self.assertEqual(len(turns), 3)  # 4 planned turns, 1 skipped


class TestRepeatedFailuresAbort(_PatchedTestCase):
    fail_when = staticmethod(lambda n: True)

    def test_consecutive_failures_raise(self):
        conv = Conversation(topic="t", agent_specs=SPECS[:2], strategy_name="round_robin")
        conv.set_max_rounds(3)
        with self.assertRaisesRegex(RuntimeError, "consecutive turns failed"):
            list(conv.run())


class TestTranscript(_PatchedTestCase):

    def test_persona_config_includes_trait(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            logger = TranscriptLogger(path)
            conv = Conversation(topic="t", agent_specs=SPECS[1:2], strategy_name="round_robin",
                                transcript_logger=logger)
            conv.set_max_rounds(1)
            list(conv.run())
            logger.close()

            with open(path, encoding="utf-8") as f:
                entries = [json.loads(line) for line in f]

        agent_entries = [e for e in entries if e["speaker_type"] == "agent"]
        self.assertEqual(agent_entries[0]["persona_config"],
                         {"temperament": "dreamer", "expertise": "airesearcher", "trait": "focused"})


class TestMentions(unittest.TestCase):

    def test_whole_token_match(self):
        self.assertTrue(_mentions("@cynical_engineer1 you're wrong", "cynical_engineer1"))
        self.assertTrue(_mentions("ask the Engineer.", "engineer"))

    def test_no_prefix_or_partial_match(self):
        self.assertFalse(_mentions("@cynical_engineer10 hi", "cynical_engineer1"))
        self.assertFalse(_mentions("engineering is hard", "engineer"))
        self.assertFalse(_mentions("anything", ""))


if __name__ == "__main__":
    unittest.main()
