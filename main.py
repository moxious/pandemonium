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
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        Config.validate()
        
        # Create conversation
        conversation = Conversation(args.topic)
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

