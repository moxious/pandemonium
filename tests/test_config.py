"""
Tests for create_chat_model provider kwargs.

Patches the provider classes so no API keys or network access are needed.
"""

import unittest
from unittest.mock import patch
from pandemonium.config import Config, create_chat_model


class TestOpenAIModelKwargs(unittest.TestCase):

    def setUp(self):
        patcher = patch.multiple(Config, OPENAI_API_KEY="test-key-not-real",
                                 OPENAI_REASONING_EFFORT="medium")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _kwargs(self, **call_kwargs):
        with patch("langchain_openai.ChatOpenAI") as chat_openai:
            create_chat_model(provider="openai", **call_kwargs)
        return chat_openai.call_args.kwargs

    def test_default_model_comes_from_config(self):
        # The code default is gpt-6.2-astra, but a local .env may override it.
        self.assertEqual(self._kwargs()["model"], Config.OPENAI_MODEL)

    def test_reasoning_effort_sent_without_temperature(self):
        kwargs = self._kwargs(temperature=0.5)
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertNotIn("temperature", kwargs)

    def test_empty_reasoning_effort_sends_temperature(self):
        with patch.object(Config, "OPENAI_REASONING_EFFORT", ""):
            kwargs = self._kwargs(model="gpt-4o", temperature=0.5)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
