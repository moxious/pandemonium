# Pandemonium

A conversational agent framework built with LangChain and LangGraph that features multiple AI personas engaging in round-robin discussions on any topic.

## Features

- **Multiple AI Personas**: Agents assembled from temperament x expertise x trait combinations
- **LangGraph Orchestration**: Conversation flow driven by a LangGraph StateGraph
- **Pluggable Turn Strategies**: Round-robin, broker-mediated, or stochastic speaker selection
- **Broker Agent**: Facilitates discussion, summarizes, and focuses conversation
- **Evaluator Agent**: Independent post-conversation synthesis against user-defined criteria
- **JSONL Transcript Logger**: Structured conversation logs with speaker metadata
- **Flexible Configuration**: Environment-based configuration with dotenv support

## Personalities

See `personas.json`; each agent is a combination of a "temperament", an "expertise", and a "trait". By default, we start with 5 random conversational participants.

**Temperaments:** cynic, dreamer, optimist, detective, empath, sunny, focused, nononsense, theory, pragmatic, power, creative, scamp, helper, wellactually, cautious, bland, dad, mom, edgy, chaos

**Expertise:** improv, politico, storyteller, comic, airesearcher, writer, PMM, devadvocate, security, hacker, engineer, engineering_manager, intern, legal, support_engineer, designer, data_analyst, people_ops, csm, account_executive, sdr, marketing, executive, productmanager, freemason

**Traits:** generalist, focus, focused, yesand, chaos, detached, edgy, socratic, devilsadvocate, visionary, realist, meta

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the environment template and add your OpenAI API key:

```bash
cp .env.template .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
TEMPERATURE=0.7
CONTEXT_ROUNDS=3
SUMMARY_AFTER_ROUNDS=3
```

### 3. Run the Application

```bash
python main.py -t "Your conversation topic here"
```

## Usage

### Basic Usage

```bash
python main.py -t "The future of artificial intelligence"
```

### Advanced Options

```bash
# Set number of conversation rounds
python main.py -t "Climate change solutions" --rounds 5

# Specify custom agents by temperament,expertise,trait
python main.py -t "AI ethics" --agents "helper,engineer,generalist" "cynic,security,focused"

# Choose a turn-taking strategy
python main.py -t "Remote work" --strategy round_robin
python main.py -t "Remote work" --strategy stochastic

# Save a JSONL transcript
python main.py -t "Space exploration" --transcript output.jsonl

# Set evaluation criteria
python main.py -t "Business observability" --criteria "most practical product ideas"

# List available personas, traits, and strategies
python main.py --list-personas

# Combine options
python main.py -t "Space exploration" --rounds 4 --strategy stochastic --agents "cautious,engineer,realist" "dreamer,executive,visionary" --transcript space.jsonl
```

### Command Line Arguments

| Flag | Description |
|------|-------------|
| `--topic, -t` | The conversation topic (string or path to a text file) |
| `--run` | Like `--topic` but auto-creates structured output in `output/<slug>/` |
| `--rounds, -r` | Number of conversation rounds (default: 3) |
| `--messages, -m` | Max messages for message-based strategies like `chatroom` (overrides `--rounds`) |
| `--criteria, -c` | Evaluation criteria for the final synthesis |
| `--agents` | Agent specs as `temperament,expertise,trait` strings |
| `--strategy, -s` | Turn strategy: `round_robin`, `broker_mediated`, `stochastic`, `chatroom` (default: `broker_mediated`) |
| `--config` | Path to JSON config file with agent/model specifications (mutually exclusive with `--agents`) |
| `--transcript, -o` | Path for JSONL transcript output |
| `--json` | Output structured JSONL to stdout (one JSON object per line) |
| `--list-personas` | List available temperaments, expertise, traits, and strategies |
| `--broker-mode` | Broker participation: `silent` (default) or `active` |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |

### Agent Specification Format

The `--agents` flag accepts `temperament,expertise,trait` strings:

- **Fully specified**: `"helper,engineer,generalist"`
- **Random temperament**: `",engineer,generalist"`
- **Random expertise**: `"helper,,focused"`
- **Trait defaults to generalist**: `"cynic,security"`

### Structured Output (`--json`)

The `--json` flag emits one JSON object per line to stdout, making the CLI parseable by scripts and AI agents. All diagnostic output goes to stderr so stdout is pure JSONL.

```bash
# Run with structured output
python main.py --json -t "AI regulation" --rounds 2

# Discover valid persona options as JSON
python main.py --list-personas --json
```

Event types emitted to stdout:

| type | fields | when |
|------|--------|------|
| `message` | `speaker`, `content`, `timestamp` | Each conversation turn and broker intro |
| `complete` | `speaker`, `content`, `timestamp` | Final evaluation and wrap-up |
| `token_summary` | `input_tokens`, `output_tokens`, `total_tokens` | After conversation completes |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error (missing API key, invalid agent spec, bad config file) |
| 2 | Runtime error (unexpected exception during conversation) |
| 130 | Keyboard interrupt |

## Turn-Taking Strategies

| Strategy | Behavior |
|----------|----------|
| `round_robin` | Fixed order, every agent speaks once per round. Broker never interjects. |
| `broker_mediated` | All agents speak once per round. Broker interjects every 3 agent turns to summarize/redirect. |
| `stochastic` | Shuffled order per round. Broker has ~30% chance of interjecting between turns. |

## How It Works

1. **Topic Introduction**: The Broker agent introduces the conversation topic
2. **Round Planning**: The turn strategy plans the speaker order for each round
3. **Speaker Selection**: The graph selects the next speaker from the planned order
4. **Agent Response**: The selected agent generates a response using its persona and recent context
5. **Round Check**: After each response, the graph checks if the round is complete
6. **Repeat**: Steps 3-5 loop until all rounds are complete
7. **Evaluation**: A fresh EvaluatorAgent synthesizes the conversation against the evaluation criteria

## Architecture

```
pandemonium/
├── __init__.py
├── config.py              # Configuration from .env
├── graph.py               # LangGraph StateGraph orchestrator
├── conversation.py        # Factory/facade: builds agents, runs graph
├── turn_strategies.py     # Pluggable turn-taking strategies
├── transcript.py          # JSONL transcript logger
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class (LangChain ChatOpenAI)
│   ├── broker.py          # Broker agent (facilitator)
│   ├── meta_agent.py      # Persona-loaded agents from personas.json
│   └── evaluator_agent.py # Post-conversation evaluator
main.py                    # CLI entry point
personas.json              # Temperament x expertise x trait definitions
```

### Graph Structure

The conversation is driven by a LangGraph `StateGraph` with these nodes:

```
START -> introduce_topic -> select_speaker -> agent_respond -> check_round
                                ^                                  |
                                |--- (round incomplete) -----------|
                                                                   |
                                         evaluate <-- (rounds done)|
                                            |
                                           END
```

State uses `Annotated[list[BaseMessage], operator.add]` for append-only message accumulation. Agent instances and the turn strategy are passed via `RunnableConfig`, not state (they aren't serializable).

## Transcript Format

When using `--transcript`, each line is a JSON object:

```json
{"timestamp": "2026-04-17T01:21:03.891635+00:00", "round_number": 0, "turn_number": 1, "speaker": "BrokerBobby", "speaker_type": "broker", "content": "...", "persona_config": null, "token_count": null}
{"timestamp": "2026-04-17T01:21:05.123456+00:00", "round_number": 0, "turn_number": 2, "speaker": "cynical_engineer3", "speaker_type": "agent", "content": "...", "persona_config": {"temperament": "cynic", "expertise": "engineer"}, "token_count": null}
```

The `token_count` field is reserved for future tiktoken integration.

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## License

This project is open source and available under the MIT License.
