"""
MetaAgent implementation that can load personas from a JSON file.
"""

import json
import logging
import os
from typing import Dict, Any
from .base_agent import BaseAgent
import random

conversation_agent_prompt = """
You are a participant in an online groupchat. You respond briefly and avoid multiple paragraphs of output.
You may use abbreviations and slang, like you're on an internet chat. Avoid emoji.

If you choose to respond to a user, use @Username to reference them. Never respond to more than
one person.

Always stay on the topic of the chatroom or you will be removed.

In your conversational style, half the time or more, make your own points. But also,
sometimes, respond to other users in the chat to keep them engaged. Remember, this is
a group conversation with other users!
If you choose to respond by @Username who said whatever you're responding to.
Never respond to more than one person at a time.

Do not output your name or any other "chat formatting", that will be handled for you.

You may choose a point of view and defend it. Try to make focused points, avoid generating
long laundry lists of points or ideas. You are allowed to disagree with other users, please do so
respectfully.
"""

_personas_cache: Dict[str, Any] = {}


def _load_personas(personas_file: str = "personas.json") -> Dict[str, Any]:
    """Load personas from the JSON file, caching the result."""
    if personas_file in _personas_cache:
        return _personas_cache[personas_file]

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    personas_path = os.path.join(project_root, personas_file)

    if not os.path.exists(personas_path):
        raise FileNotFoundError(f"Personas file not found at {personas_path}")

    try:
        with open(personas_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in personas file: {e}")

    _personas_cache[personas_file] = data
    return data


def validate_spec(temperament: str = None, expertise: str = None, trait: str = "generalist",
                  personas_file: str = "personas.json"):
    """Validate persona spec fields against personas.json without constructing an agent."""
    personas = _load_personas(personas_file)
    if temperament and temperament not in personas["temperaments"]:
        raise ValueError(f"Temperament '{temperament}' not found. Available: {list(personas['temperaments'].keys())}")
    if expertise and expertise not in personas["expertise"]:
        raise ValueError(f"Expertise '{expertise}' not found. Available: {list(personas['expertise'].keys())}")
    if trait and trait not in personas["traits"]:
        raise ValueError(f"Trait '{trait}' not found. Available: {list(personas['traits'].keys())}")


class MetaAgent(BaseAgent):
    """A meta agent that can take on different personas loaded from a JSON file."""

    def __init__(self, temperament: str = None, expertise: str = None, trait: str = "generalist",
                 personas_file: str = "personas.json", model=None, provider=None):
        personas = _load_personas(personas_file)

        if not temperament:
            temperament = random.choice(list(personas["temperaments"].keys()))
        if not expertise:
            expertise = random.choice(list(personas["expertise"].keys()))
        if not trait:
            trait = random.choice(list(personas["traits"].keys()))

        self.temperament = temperament
        self.expertise = expertise

        validate_spec(temperament, expertise, trait, personas_file)

        name = personas["temperaments"][temperament]["description"] + "_" + expertise + "%d" % random.randint(1, 10)
        temperament_description = personas["temperaments"][temperament]['persona']
        expertise_description = personas["expertise"][expertise]['persona']
        trait_description = personas["traits"][trait]

        persona = f"""{conversation_agent_prompt}

        {temperament_description}

        {expertise_description}

        Your conversational traits control how you interact with others in the chatroom. Your trait behavior is:

        {trait}: {trait_description}
        """

        super().__init__(name=name, persona=persona, model=model, provider=provider)
        self.logger = logging.getLogger(f"pandemonium.agents.meta.{name}")
