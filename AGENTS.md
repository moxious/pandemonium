# Pandemonium — Agent Guide

This file is the shared context for all AI coding agents working on this repo.

## What This Project Is

A multi-persona AI chatroom simulator. LLM-powered agents with distinct personalities discuss a topic in rounds, orchestrated by a LangGraph StateGraph. A broker agent facilitates, and an evaluator agent synthesizes the outcome at the end.

## Key Architecture Decisions

- **LangGraph drives orchestration.** The conversation loop lives in `graph.py` as a compiled StateGraph, not in application code. `conversation.py` is a factory that builds agents and invokes the graph.
- **Round-based context windowing.** Agent memory is a `list[BaseMessage]` managed by graph state. Before each turn, `agent_respond` builds a context window from the last `CONTEXT_ROUNDS` rounds (default 3), always pinning the topic introduction at the top. For long conversations (≥`SUMMARY_AFTER_ROUNDS` rounds), a broker-generated summary of earlier discussion is prepended so agents don't lose the thread. Windowing is done in `graph.py`; `base_agent._create_messages()` trusts whatever it receives.
- **Append-only state.** `ConversationState.messages` uses `Annotated[list, operator.add]` so graph nodes return new messages to append, never replace the list.
- **Agents passed via config, not state.** Agent objects (with LLM clients) go in `RunnableConfig["configurable"]`, not in `ConversationState`. State must remain serializable.
- **Pluggable turn strategies.** Who speaks next is controlled by a `TurnStrategy` (in `turn_strategies.py`), not by the broker. The broker is a participant, not a controller.

## File Map

| File | Role |
|------|------|
| `graph.py` | LangGraph StateGraph — the orchestration heart |
| `conversation.py` | Factory: builds agents, strategy, graph; exposes `run()` iterator |
| `turn_strategies.py` | `TurnStrategy` ABC + RoundRobin, BrokerMediated, Stochastic |
| `transcript.py` | JSONL transcript logger |
| `agents/base_agent.py` | Abstract base with LLM, memory, and message construction |
| `agents/broker.py` | Facilitator persona agent |
| `agents/meta_agent.py` | Persona-loaded from `personas.json` (temperament x expertise x trait) |
| `agents/evaluator_agent.py` | Post-conversation evaluator |
| `config.py` | Env-based config (API key, model, temperature, context rounds) |
| `personas.json` | Persona definitions |
| `main.py` | CLI entry point |

## CLI for Agent Consumption

The CLI (`main.py`) supports a `--json` flag for structured output, making it callable by AI agent orchestrators. All errors go to stderr regardless of mode. Stdout is pure JSONL when `--json` is active.

### Discovery: enumerate valid options

```bash
python main.py --list-personas --json
```

Returns a single JSON object:

```json
{
  "temperaments": ["cynic", "dreamer", "optimist", ...],
  "expertise": ["engineer", "security", "airesearcher", ...],
  "traits": ["generalist", "focused", "chaos", ...],
  "strategies": ["round_robin", "broker_mediated", "stochastic", "chatroom"]
}
```

Use these values to construct valid `--agents` and `--strategy` arguments.

### Running a conversation with structured output

```bash
python main.py --json \
  -t "Should we regulate AI?" \
  --rounds 2 \
  --strategy round_robin \
  --agents "cynic,engineer,generalist" "dreamer,airesearcher,focused"
```

Stdout emits one JSON object per line. Three event types:

**Message event** (conversation turns and broker intro):
```json
{"type": "message", "speaker": "cynical_engineer3", "content": "cynical_engineer3: ...", "timestamp": "2026-04-20T12:00:00+00:00"}
```

**Complete event** (final evaluation and wrap-up):
```json
{"type": "complete", "speaker": "system", "content": "--- Conversation Complete ---\n...", "timestamp": "2026-04-20T12:01:00+00:00"}
```

**Token summary** (emitted last, only if tokens were tracked):
```json
{"type": "token_summary", "input_tokens": 4500, "output_tokens": 1200, "total_tokens": 5700}
```

### Agent spec format

`--agents` accepts space-separated `"temperament,expertise,trait"` strings. Each field can be empty for random selection:

- `"cynic,engineer,generalist"` — fully specified
- `",engineer,generalist"` — random temperament
- `"cynic,,focused"` — random expertise
- `"cynic,engineer"` — trait defaults to `generalist`

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error (missing API key, invalid agent spec, bad config file) |
| 2 | Runtime error (unexpected exception during conversation) |
| 130 | Keyboard interrupt |

### Complete flag reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--topic, -t` | string | — | Conversation topic (string or path to a text file) |
| `--run` | string | — | Like `--topic` but auto-creates structured output in `output/<slug>/` |
| `--rounds, -r` | int | 3 | Number of conversation rounds |
| `--messages, -m` | int | 0 | Max messages for message-based strategies (overrides `--rounds`) |
| `--criteria, -c` | string | `"pick most interesting or important issues"` | Evaluation criteria for the final synthesis |
| `--agents` | string[] | 5 random | Agent specs as `"temperament,expertise,trait"` strings |
| `--strategy, -s` | choice | `broker_mediated` | `round_robin`, `broker_mediated`, `stochastic`, `chatroom` |
| `--config` | path | — | JSON config file with agent/model specs (mutually exclusive with `--agents`) |
| `--transcript, -o` | path | — | Path for detailed JSONL transcript output |
| `--json` | flag | off | Structured JSONL output to stdout |
| `--list-personas` | flag | — | List valid temperaments, expertise, traits, strategies |
| `--broker-mode` | choice | `silent` | `silent` (intro + summaries) or `active` (speaks during rounds) |
| `--log-level` | choice | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Typical agent workflow

1. Discover options: `python main.py --list-personas --json`
2. Parse JSON, select temperaments/expertise/traits for the use case
3. Run: `python main.py --json -t "topic" --rounds 2 --agents "spec1" "spec2"`
4. Read stdout line by line, parse each as JSON
5. Check exit code: 0 = success, non-zero = check stderr for error message

## Conventions

- Graph node functions take `(state: ConversationState, config: RunnableConfig)` and return a dict of state updates.
- Turn strategies implement `plan_round()` and `is_round_complete()`.
- Transcript entries are dataclasses serialized to JSONL with `json.dumps(asdict(...))`.
- The `token_count` field in transcript entries is `None` until tiktoken is wired in.
## Testing

Run the end-to-end tests after any major modification to verify the conversation pipeline still works:

```
.venv/bin/python -m unittest tests.test_e2e -v
```

The e2e tests mock `create_chat_model` at both import sites (`agents/base_agent.py` and `agents/evaluator_agent.py`) so no API keys or network access are needed. They exercise the full LangGraph pipeline: topic introduction, speaker selection, agent responses, round management, evaluation, and token tracking.

When adding new tests, follow the same pattern — patch `create_chat_model` where it's imported, not just in `config.py`.
