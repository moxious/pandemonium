"""
JSONL transcript logger for Pandemonium conversations.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any


@dataclass
class TranscriptEntry:
    """A single entry in the conversation transcript."""
    timestamp: str
    round_number: int
    turn_number: int
    speaker: str
    speaker_type: str  # "agent" | "broker" | "evaluator" | "system"
    content: str
    persona_config: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None  # Ready for tiktoken integration


class TranscriptLogger:
    """Writes conversation transcript entries to a JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._turn_counter = 0
        self._file = open(output_path, "a", encoding="utf-8")

    def log_entry(
        self,
        round_number: int,
        speaker: str,
        speaker_type: str,
        content: str,
        persona_config: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
    ):
        """Log a transcript entry and write it to the JSONL file."""
        self._turn_counter += 1
        entry = TranscriptEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            round_number=round_number,
            turn_number=self._turn_counter,
            speaker=speaker,
            speaker_type=speaker_type,
            content=content,
            persona_config=persona_config,
            token_count=token_count,
        )
        self._file.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        """Close the file handle."""
        self._file.close()
