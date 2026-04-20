# TODO

## Context window limits in long debates

Per-agent context is bounded by the sliding window (CONTEXT_ROUNDS / CONTEXT_MESSAGES), but two paths feed the full unwindowed message history to an LLM and will blow context limits in long debates:

1. **Summarizer** (`graph.py:_generate_summary`) — joins all messages into a single string and sends to the broker's LLM. First thing to break.
2. **Evaluator** (`graph.py:evaluate`) — dumps the full conversation history into the evaluator's context at the end.

The orchestrator's `state["messages"]` list also grows without bound in memory (append-only), which is a secondary concern.

Fix: window or chunk the input to both the summarizer and evaluator, or use incremental summarization (summarize-the-summary) instead of re-reading everything.
