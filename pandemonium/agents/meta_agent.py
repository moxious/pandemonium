"""
MetaAgent implementation that can load personas from a JSON file.
"""

import json
import logging
import os
from typing import Dict, Any
from .base_agent import BaseAgent
import random

class MetaAgent(BaseAgent):
    """A meta agent that can take on different personas loaded from a JSON file."""
    
    def __init__(self, temperament: str = None, expertise: str = None, personas_file: str = "personas.json"):
        """
        Initialize the MetaAgent with a specific persona.
        
        Args:
            persona_key: The key in the personas.json file to load
            temperament: The temperament of the agent
            expertise: The expertise of the agent
            personas_file: Path to the personas JSON file (default: "personas.json")
        """
        # Load personas from JSON file
        personas = self._load_personas(personas_file)
        self.temperament = temperament
        self.expertise = expertise

        if not temperament or temperament is None:
            temperament = random.choice(list(personas["temperments"].keys()))
        if not expertise or expertise is None:
            expertise = random.choice(list(personas["expertise"].keys()))

        if temperament not in personas["temperments"]:
            raise ValueError(f"Temperament key '{temperament}' not found in {personas_file}. Available personas: {list(personas["temperments"].keys())}")

        if expertise not in personas["expertise"]:
            raise ValueError(f"Expertise key '{expertise}' not found in {personas_file}. Available personas: {list(personas["expertise"].keys())}")

        name = personas["temperments"][temperament]["name"] + "_" + expertise
        temp = personas["temperments"][temperament]['persona']
        expertise = personas["expertise"][expertise]['persona']

        persona = temp + "\n\n" + expertise
        
        # Initialize with the loaded persona
        super().__init__(name, persona)
        self.logger = logging.getLogger(f"pandemonium.agents.meta.{temperament}_{expertise}")
    
    def _load_personas(self, personas_file: str) -> Dict[str, Any]:
        """Load personas from the JSON file."""
        # Try to find the personas file in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        personas_path = os.path.join(project_root, personas_file)
        
        if not os.path.exists(personas_path):
            raise FileNotFoundError(f"Personas file not found at {personas_path}")
        
        try:
            with open(personas_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in personas file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading personas file: {e}")
    
    def respond(self, topic: str) -> str:
        """Generate a response based on the loaded persona."""
        self.logger.info(f"Generating {self.name} response for topic: {topic}")
        messages = self._create_messages(topic)
        response = self.llm.invoke(messages)
        
        # Add response to memory for future context
        self._add_response_to_memory(response.content)
        
        return response.content
    
    @classmethod
    def get_available_personas(cls, personas_file: str = "personas.json") -> list:
        """Get a list of available persona keys from the JSON file."""
        try:
            # Try to find the personas file in the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            personas_path = os.path.join(project_root, personas_file)
            
            if not os.path.exists(personas_path):
                return []
            
            with open(personas_path, 'r', encoding='utf-8') as f:
                personas = json.load(f)
                return list(personas["temperments"].keys())
        except Exception as e:
            logging.getLogger("pandemonium.agents.meta").error(f"Error loading available personas: {e}")
            return []
