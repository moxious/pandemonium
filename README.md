# Pandemonium

A conversational agent framework built with LangChain that features multiple AI personas engaging in round-robin discussions on any topic.

## Features

- **Multiple AI Personas**: Numerous conversational agents & combinations
- **Broker Agent**: Manages turn-taking and conversation flow
- **Progressive Context**: Each agent only sees new conversation context, building state over time
- **Flexible Configuration**: Environment-based configuration with dotenv support
- **Interactive & Automatic Modes**: Run conversations automatically or step through manually

## Personalities

See `personas.json`; each agent is a combination of a "temperament" (cynical, dreamy, questioning) and an "expertise"
(engineer, legal, marketing, etc).  By default, we start with 5 random conversational participants, who round-robin
discuss the topic.

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

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. Run the Application

```bash
python main.py "Your conversation topic here"
```

## Usage

### Basic Usage

Start a conversation on any topic:

```bash
python main.py "The future of artificial intelligence"
```

### Advanced Options

```bash
# Set number of conversation rounds
python main.py "Climate change solutions" --rounds 5

# Interactive mode (step through each turn)
python main.py "Remote work vs office work" --interactive

# Combine options
python main.py "Space exploration" --rounds 4 --interactive
```

### Command Line Arguments

- `topic`: The conversation topic (required)
- `--rounds, -r`: Number of conversation rounds (default: 3)
- `--interactive, -i`: Run in interactive mode for manual turn progression

## How It Works

1. **Topic Introduction**: The Broker agent introduces the conversation topic
2. **Round-Robin Discussion**: Agents take turns responding in a fixed order:
   - The Cynic
   - The Dreamer  
   - The Cautious
3. **Progressive Context**: Each agent only receives conversation context they haven't seen before
4. **State Building**: Agents build understanding of the conversation over multiple rounds
5. **Conversation Conclusion**: After the specified rounds, the Broker concludes the discussion

## Example Output

```
Welcome to our conversation! Today we'll be discussing: The future of artificial intelligence

I'll be facilitating this discussion, and we have three distinct perspectives joining us:
- The Cynic, who will bring skepticism and critical thinking
- The Dreamer, who will share optimism and big-picture thinking  
- The Cautious, who will consider risks and implications carefully

Let's begin with our first round of thoughts on this topic.

==================================================

The Cynic: While everyone's excited about AI's potential, I can't help but wonder if we're getting ahead of ourselves. What about the massive energy consumption? The bias in training data? The fact that we're essentially creating black boxes that even their creators don't fully understand?

------------------------------

The Dreamer: I see AI as humanity's greatest leap forward! Imagine personalized education for every child, medical breakthroughs happening in real-time, and creative collaborations between humans and machines that we can't even conceive of yet. We're on the brink of solving problems that have plagued us for centuries!

------------------------------

The Cautious: Both perspectives raise valid points. Before we rush forward, we need to establish robust ethical frameworks, ensure equitable access, and create safeguards against misuse. The potential is enormous, but so are the risks if we don't proceed thoughtfully.
```

## Architecture

```
pandemonium/
├── __init__.py
├── config.py              # Configuration management
├── conversation.py        # Conversation orchestration
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   ├── broker.py          # Broker agent
│   ├── cynic.py           # Cynic persona
│   ├── dreamer.py         # Dreamer persona
│   └── cautious.py        # Cautious persona
├── main.py                # CLI application
├── requirements.txt       # Dependencies
├── .env.template         # Environment template
└── README.md             # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## License

This project is open source and available under the MIT License.

