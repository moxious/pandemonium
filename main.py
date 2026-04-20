#!/usr/bin/env python3
"""
Pandemonium: A conversational agent framework with multiple personas.

Usage:
    python main.py --topic "Your conversation topic here"
"""

import sys
import os
import re
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from pandemonium.config import Config
from pandemonium.conversation import Conversation
from pandemonium.turn_strategies import STRATEGIES

OUTPUT_DIR = Path(__file__).parent / "output"

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_KEYBOARD_INTERRUPT = 130

_json_mode = False


def _eprint(*args, **kwargs):
    """Print to stderr (for errors)."""
    print(*args, file=sys.stderr, **kwargs)


def _info(*args, **kwargs):
    """Print diagnostic info — to stderr in JSON mode, stdout otherwise."""
    if _json_mode:
        print(*args, file=sys.stderr, **kwargs)
    else:
        print(*args, **kwargs)


def _json_event(event_type, **fields):
    """Write a single JSONL event line to stdout."""
    print(json.dumps({"type": event_type, **fields}, ensure_ascii=False))


def _slugify(text: str) -> str:
    """Turn a topic string or filename into a filesystem-safe slug."""
    # If it looks like a file path, use the stem
    if os.path.isfile(text):
        text = Path(text).stem
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80] or "untitled"


class TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    # Forward any other attribute access to the original stream
    def __getattr__(self, name):
        return getattr(self.original, name)


def main():
    """Main entry point for the Pandemonium application."""
    parser = argparse.ArgumentParser(
        description="Pandemonium: A conversational agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py -t "The future of artificial intelligence"
    python main.py -t "Climate change solutions" --rounds 5
    python main.py -t "AI ethics" --agents "helper,engineer,generalist" "cynic,security,focused"
    python main.py -t "Space exploration" --strategy stochastic --transcript out.jsonl
    python main.py -t "AI ethics" --config conversation.json
    python main.py --run freemasonry.txt --rounds 3 --config masonic.json
    python main.py --run "AI ethics" --strategy chatroom --messages 20
        """
    )

    parser.add_argument(
        "--topic", "-t",
        nargs="?",
        help="The topic for the conversation (string or path to a file)"
    )

    parser.add_argument(
        "--run",
        metavar="TOPIC_OR_FILE",
        help="Run mode: like --topic but auto-creates structured output in output/<slug>/"
    )

    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of conversation rounds (default: 3)"
    )

    parser.add_argument(
        "--criteria", "-c",
        default="pick most interesting or important issues",
        help="The criteria for the broker to evaluate the conversation"
    )

    parser.add_argument(
        "--agents",
        nargs="*",
        help="Specify agents as 'temperament,expertise,trait' (e.g., 'helper,engineer,generalist')"
    )

    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()) + ["chatroom"],
        default="broker_mediated",
        help="Turn-taking strategy (default: broker_mediated)"
    )

    parser.add_argument(
        "--messages", "-m",
        type=int,
        default=0,
        help="Max messages for message-based strategies like 'chatroom' (overrides --rounds)"
    )

    parser.add_argument(
        "--transcript", "-o",
        help="Path for JSONL transcript output"
    )

    parser.add_argument(
        "--config",
        help="Path to JSON config file with agent/model specifications"
    )

    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List available temperament and expertise options from personas.json"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSONL to stdout (one JSON object per line)"
    )

    parser.add_argument(
        "--broker-mode",
        choices=["silent", "active"],
        default="silent",
        help="Broker participation mode: 'silent' (intro + summaries only) or 'active' (speaks during rounds) (default: silent)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )

    args = parser.parse_args()

    global _json_mode
    _json_mode = args.json

    try:
        # Setup logging with command line log level
        Config.LOG_LEVEL = args.log_level
        Config.setup_logging()

        # Handle --list-personas flag
        if args.list_personas:
            _list_personas()
            return

        # --run is shorthand for --topic + auto-logging
        run_log_file = None
        tee_stdout = None
        tee_stderr = None
        if args.run:
            if args.topic:
                parser.error("--run and --topic are mutually exclusive")
            args.topic = args.run

        # Validate that topic is provided when not listing personas
        if not args.topic:
            parser.error("topic is required when not using --list-personas")

        # If topic is a file path, read its contents
        topic = args.topic
        if os.path.isfile(topic):
            with open(topic, 'r', encoding='utf-8') as f:
                topic = f.read().strip()
            _info(f"Loaded topic from file: {args.topic}")
        args.topic = topic

        # Set up auto-logging for --run mode
        if args.run:
            slug = _slugify(args.run)
            topic_dir = OUTPUT_DIR / slug
            topic_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            jsonl_path = topic_dir / f"{stamp}.jsonl"
            log_path = topic_dir / f"{stamp}.log"

            # Auto-set transcript path (user can still override with --transcript)
            if not args.transcript:
                args.transcript = str(jsonl_path)

            # Tee console output to the .log file
            run_log_file = open(log_path, "w", encoding="utf-8")
            tee_stdout = TeeStream(sys.stdout, run_log_file)
            tee_stderr = TeeStream(sys.stderr, run_log_file)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            # Update latest symlinks
            for name, target in [("latest.jsonl", jsonl_path.name), ("latest.log", log_path.name)]:
                link = topic_dir / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)

            _info(f"Run mode: output → {topic_dir}/")
            _info(f"  transcript: {jsonl_path.name}")
            _info(f"  console:    {log_path.name}")

        # Validate configuration
        Config.validate()

        # Parse agent and broker specifications
        agent_specs = None
        broker_config = None

        if args.config and args.agents:
            parser.error("--config and --agents are mutually exclusive")

        if args.config:
            agent_specs, broker_config = _load_config_file(args.config)
        elif args.agents:
            agent_specs = _parse_agent_specs(args.agents)

        # Set up transcript logger if requested
        transcript_logger = None
        if args.transcript:
            from pandemonium.transcript import TranscriptLogger
            transcript_logger = TranscriptLogger(args.transcript)

        # Create and run conversation
        conversation = Conversation(
            topic=args.topic,
            agent_specs=agent_specs,
            evaluation_criteria=args.criteria,
            strategy_name=args.strategy,
            transcript_logger=transcript_logger,
            broker_config=broker_config,
            broker_mode=args.broker_mode,
        )
        conversation.set_max_rounds(args.rounds)

        if args.messages > 0:
            conversation.set_max_messages(args.messages)
        elif args.strategy == "chatroom":
            conversation.set_max_messages(30)

        _info(f"Strategy: {args.strategy}")
        _info("=" * 50)

        for event in conversation.run():
            if args.json:
                _json_event(
                    event["type"],
                    speaker=event.get("speaker", ""),
                    content=event["content"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            else:
                if event["type"] == "message":
                    print(f"\n{event['content']}")
                elif event["type"] == "complete":
                    print(f"\n{event['content']}")

        # Print token usage summary
        if conversation.token_tracker.total_tokens > 0:
            t = conversation.token_tracker
            if args.json:
                _json_event(
                    "token_summary",
                    input_tokens=t.total_input_tokens,
                    output_tokens=t.total_output_tokens,
                    total_tokens=t.total_tokens,
                )
            else:
                print(f"\nToken usage: {t.total_input_tokens:,} input + {t.total_output_tokens:,} output = {t.total_tokens:,} total")

        if transcript_logger:
            transcript_logger.close()
            _info(f"\nTranscript saved to: {args.transcript}")

    except ValueError as e:
        _eprint(f"Configuration error: {e}")
        _eprint("Please make sure you have set the required API keys in your .env file.")
        sys.exit(EXIT_CONFIG_ERROR)
    except KeyboardInterrupt:
        _eprint("\n\nConversation interrupted. Goodbye!")
        sys.exit(EXIT_KEYBOARD_INTERRUPT)
    except Exception as e:
        _eprint(f"Unexpected error: {e}")
        sys.exit(EXIT_RUNTIME_ERROR)
    finally:
        # Restore original streams and close the log file
        if tee_stdout is not None:
            sys.stdout = tee_stdout.original
        if tee_stderr is not None:
            sys.stderr = tee_stderr.original
        if run_log_file is not None:
            run_log_file.close()


def _list_personas():
    """List available temperament and expertise options."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    personas_path = os.path.join(project_root, "personas.json")

    try:
        with open(personas_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)

        if _json_mode:
            output = {
                "temperaments": list(personas["temperaments"].keys()),
                "expertise": list(personas["expertise"].keys()),
                "traits": list(personas["traits"].keys()),
                "strategies": list(STRATEGIES.keys()) + ["chatroom"],
            }
            print(json.dumps(output, ensure_ascii=False))
            return

        print("Available Temperaments:")
        for key, value in personas["temperaments"].items():
            label = value.get('name', value.get('description', key))
            print(f"  {key}: {label}")

        print("\nAvailable Expertise:")
        for key, value in personas["expertise"].items():
            label = value.get('name', value.get('description', key))
            print(f"  {key}: {label}")

        print("\nAvailable Traits:")
        for key in personas["traits"]:
            print(f"  {key}")

        print("\nAvailable Strategies:")
        for name in STRATEGIES:
            print(f"  {name}")

        print("\nExample usage:")
        print("  python main.py -t 'AI ethics' --agents 'helper,engineer,generalist' 'cynic,security,focused'")

    except Exception as e:
        _eprint(f"Error loading personas: {e}")


def _load_config_file(config_path):
    """Load agent and broker configuration from a JSON file."""
    if not os.path.isfile(config_path):
        _eprint(f"Config file not found: {config_path}")
        sys.exit(EXIT_CONFIG_ERROR)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        _eprint(f"Invalid JSON in config file: {e}")
        sys.exit(EXIT_CONFIG_ERROR)

    # Apply defaults to Config if provided
    defaults = config_data.get("defaults", {})
    if "provider" in defaults:
        Config.DEFAULT_PROVIDER = defaults["provider"]
    if "model" in defaults:
        # Set the appropriate model based on provider
        provider = defaults.get("provider", Config.DEFAULT_PROVIDER)
        if provider == "anthropic":
            Config.ANTHROPIC_MODEL = defaults["model"]
        else:
            Config.OPENAI_MODEL = defaults["model"]
    if "temperature" in defaults:
        Config.TEMPERATURE = float(defaults["temperature"])

    # Parse agent specs as list of dicts
    agent_specs = config_data.get("agents", None)
    if agent_specs:
        from pandemonium.agents.meta_agent import validate_spec
        _info("Validating agent specifications from config...")
        for i, spec in enumerate(agent_specs):
            temperament = spec.get("temperament")
            expertise = spec.get("expertise")
            trait = spec.get("trait", "generalist")
            provider = spec.get("provider")
            model = spec.get("model")
            validate_spec(temperament=temperament, expertise=expertise, trait=trait)
            _info(f"  Agent {i+1}: {temperament or 'random'},{expertise or 'random'},{trait} [{provider or 'default'}:{model or 'default'}]")

    # Parse broker config
    broker_config = config_data.get("broker", None)

    return agent_specs, broker_config


def _parse_agent_specs(agent_args):
    """Parse and validate agent specification strings."""
    from pandemonium.agents.meta_agent import validate_spec

    agent_specs = []
    _info("Validating agent specifications...")
    for i, agent_spec in enumerate(agent_args):
        try:
            parts = agent_spec.lower().split(',')
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid spec '{agent_spec}'. Expected: 'temperament,expertise,trait'"
                )

            temperament = parts[0].strip() or None
            expertise = parts[1].strip() or None
            trait = parts[2].strip() if len(parts) > 2 else "generalist"

            validate_spec(temperament=temperament, expertise=expertise, trait=trait)
            agent_specs.append((temperament, expertise, trait))
            _info(f"  Agent {i+1}: {temperament or 'random'},{expertise or 'random'},{trait}")

        except Exception as e:
            _eprint(f"  Invalid agent specification '{agent_spec}': {e}")
            sys.exit(EXIT_CONFIG_ERROR)

    return agent_specs


if __name__ == "__main__":
    main()
