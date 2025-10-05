#!/usr/bin/env python3
"""
Pandemonium: A conversational agent framework with multiple personas.

Usage:
    python main.py "Your conversation topic here"
"""

import sys
import argparse
from pandemonium.config import Config
from pandemonium.conversation import Conversation
import re

def main():
    """Main entry point for the Pandemonium application."""
    parser = argparse.ArgumentParser(
        description="Pandemonium: A conversational agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "The future of artificial intelligence"
    python main.py "Climate change solutions" --rounds 5
    python main.py "Remote work vs office work" --rounds 2
        """
    )
    
    parser.add_argument(
        "topic",
        nargs="?",
        help="The topic for the conversation"
    )
    
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of conversation rounds (default: 3)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode where you can continue the conversation"
    )
    
    parser.add_argument(
        "--agents",
        nargs="*",
        help="Specify agents by temperament,expertise combinations (e.g., 'helper,engineer' 'cynic,security'). Use ',' for random temperament or expertise."
    )
    
    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List available temperament and expertise options from personas.json"
    )
    
    args = parser.parse_args()
    
    try:
        # Handle --list-personas flag
        if args.list_personas:
            from pandemonium.agents.meta_agent import MetaAgent
            try:
                # Load personas to show available options
                import json
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                personas_path = os.path.join(project_root, "personas.json")
                
                with open(personas_path, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                
                print("Available Temperaments:")
                for key, value in personas["temperments"].items():
                    print(f"  {key}: {value['name']}")
                
                print("\nAvailable Expertise:")
                for key, value in personas["expertise"].items():
                    print(f"  {key}: {value['name']}")
                
                print("\nExample usage:")
                print("  python main.py 'AI ethics' --agents 'helper,engineer' 'cynic,security'")
                print("  python main.py 'Climate change' --agents ',marketing' 'dreamer,' 'focused,legal'")
                return
            except Exception as e:
                print(f"Error loading personas: {e}")
                return
        
        # Validate that topic is provided when not listing personas
        if not args.topic:
            parser.error("topic is required when not using --list-personas")
        
        # Validate configuration
        Config.validate()
        
        # Validate agent specifications if provided
        agent_specs = []
        if args.agents:
            from pandemonium.agents.meta_agent import MetaAgent
            print("Validating agent specifications...")
            for i, agent_spec in enumerate(args.agents):
                try:
                    # Parse the agent specification
                    parts = re.split(r'[,:]', agent_spec.lower())
                    if len(parts) != 2:
                        raise ValueError(f"Invalid agent specification '{agent_spec}'. Expected format: 'temperament,expertise' or 'temperament:expertise'")
                    
                    temperament, expertise = parts
                    
                    # Convert empty strings to None for random selection
                    temperament = temperament.strip() if temperament.strip() else None
                    expertise = expertise.strip() if expertise.strip() else None
                    
                    # Validate by creating a MetaAgent (this will throw an error if keys are invalid)
                    test_agent = MetaAgent(temperament=temperament, expertise=expertise)
                    agent_specs.append((temperament, expertise))
                    print(f"  ✓ Agent {i+1}: {temperament or 'random'},{expertise or 'random'}")
                    
                except Exception as e:
                    print(f"  ✗ Invalid agent specification '{agent_spec}': {e}")
                    sys.exit(1)
        
        # Create conversation
        conversation = Conversation(args.topic, agent_specs=agent_specs)
        conversation.set_max_rounds(args.rounds)
        
        # Start conversation
        print(conversation.start_conversation())
        print("\n" + "="*50 + "\n")
        
        if args.interactive:
            # Interactive mode
            while True:
                try:
                    user_input = input("Press Enter for next turn (or 'quit' to exit): ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    response = conversation.next_turn()
                    if "Conversation Complete" in response:
                        print(response)
                        break
                    
                    print(response)
                    print("\n" + "-"*30 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\nConversation interrupted. Goodbye!")
                    break
        else:
            # Automatic mode - run all rounds
            for turn in range(args.rounds * 3):  # 3 agents per round
                response = conversation.next_turn()
                print()
                if "Conversation Complete" in response:
                    print(response)
                    break
                
                print(response)
                
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please make sure you have set OPENAI_API_KEY in your .env file.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nConversation interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

