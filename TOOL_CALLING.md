# Tool Calling in Pandemonium

This document describes the tool calling functionality that has been added to Pandemonium agents, allowing them to use external tools like web search to enhance their responses.

## Overview

The tool calling system extends the existing `BaseAgent` class to support LangChain agents with tool integration while maintaining full backward compatibility. Agents can now:

- Search the web for current information
- Verify facts and claims
- Provide up-to-date context
- Use tools selectively based on their role

## Architecture

### Core Components

1. **BaseTool**: Abstract base class for all tools
2. **WebSearchTool**: DuckDuckGo search implementation
3. **Enhanced BaseAgent**: Supports both traditional LLM and LangChain agent modes
4. **ModelCompatibilityChecker**: Validates model capabilities and filters unsupported parameters
5. **ToolManager**: Handles tool registration and global tool assignment
6. **Configuration**: Tool management via environment variables

### Tool Integration

Tools are integrated using LangChain's agent framework:
- **ReAct Agent**: Uses reasoning and acting pattern for tool usage
- **Automatic Tool Selection**: Agents automatically decide when to use tools based on context
- **Model Compatibility**: Dynamic parameter filtering based on model capabilities
- **Fallback**: Agents indicate when tool information is unavailable

### Model Compatibility

The system supports OpenAI GPT-5 and GPT-5-mini series with automatic parameter filtering:
- **Parameter Validation**: Checks model capabilities before API calls
- **Graceful Degradation**: Removes unsupported parameters (like 'stop') automatically
- **Error Prevention**: Prevents 400 errors from unsupported parameters

## Configuration

### Environment Variables

```bash
# Enable tool calling
ENABLE_TOOLS=true

# Specify allowed tools (comma-separated)
ALLOWED_TOOLS=web_search

# Tool timeout in seconds
TOOL_TIMEOUT=30
```

### Tool Assignment Strategy

All agents derived from MetaAgent have uniform tool access:

- **Universal Access**: All MetaAgent instances get the same tool set
- **Global Configuration**: Tool availability controlled via environment variables
- **No Role-Based Restrictions**: No agent-specific tool limitations
- **Independent Tool Calls**: Each tool call is independent with no caching

## Usage

### Basic Usage

```python
from pandemonium.config import Config
from pandemonium.agents.meta_agent import MetaAgent

# Enable tools
Config.ENABLE_TOOLS = True

# Create agent with tools
agent = MetaAgent(temperament="cynic", expertise="engineer")

# Agent will automatically use tools when appropriate
response = agent.respond("What are the latest AI trends?")
```

### Running Conversations with Tools

```python
from pandemonium.conversation import Conversation

# Set environment variables
import os
os.environ["ENABLE_TOOLS"] = "true"

# Create conversation
conversation = Conversation(
    topic="AI ethics",
    agent_specs=[("cynic", "engineer", "generalist")]
)

# Start conversation - agents will use tools as needed
print(conversation.start_conversation())
```

## Available Tools

### WebSearchTool

Searches the web using DuckDuckGo for current information.

**Features:**
- No API key required
- Privacy-focused search
- Configurable result limits
- Error handling and fallback

**Usage:**
```python
from pandemonium.tools import WebSearchTool

tool = WebSearchTool()
results = tool.execute("artificial intelligence trends 2024")
```

## Implementation Details

### Model Compatibility System

The system includes a comprehensive model compatibility checker to prevent parameter errors:

```python
# Model capability registry
MODEL_CAPABILITIES = {
    "gpt-5": {
        "supports_stop": True,
        "supports_tools": True,
        "max_tokens": 128000
    },
    "gpt-5-mini": {
        "supports_stop": True,
        "supports_tools": True,
        "max_tokens": 128000
    }
}

# Automatic parameter filtering
def filter_parameters_for_model(model_name, parameters):
    capabilities = MODEL_CAPABILITIES.get(model_name, {})
    if not capabilities.get("supports_stop", False):
        parameters.pop("stop", None)
    return parameters
```

### Backward Compatibility

The tool calling system maintains full backward compatibility:

- **Default Behavior**: Tools disabled by default
- **Existing Code**: Works unchanged when tools disabled
- **Gradual Migration**: Can enable tools per agent type
- **Fallback**: Graceful degradation if tool setup fails
- **Model Safety**: Automatic parameter filtering prevents API errors

### Tool Selection Logic

Agents use tools based on:

1. **Configuration**: Only if `ENABLE_TOOLS=true`
2. **Agent Type**: Different tools for different roles
3. **Context**: Tools used when relevant to conversation
4. **Persona**: Tool usage instructions in agent prompts

### Error Handling

- **Tool Failures**: Agents indicate when tool information is unavailable
- **Model Compatibility**: Automatic parameter filtering prevents unsupported parameter errors
- **Network Issues**: Graceful degradation with clear error messages
- **Configuration Errors**: Clear error messages and validation
- **Timeout**: Configurable tool execution limits
- **Stop Parameter Errors**: Automatically filtered for GPT-5 series compatibility

## Examples

### Example 1: Basic Tool Usage

```python
#!/usr/bin/env python3
import os
from pandemonium.agents.meta_agent import MetaAgent

# Enable tools
os.environ["ENABLE_TOOLS"] = "true"

# Create agent
agent = MetaAgent(temperament="focused", expertise="engineer")

# Agent will use web search when appropriate
response = agent.respond("What are the latest developments in quantum computing?")
print(response)
```

### Example 2: Conversation with Tools

```python
#!/usr/bin/env python3
import os
from pandemonium.conversation import Conversation

# Enable tools
os.environ["ENABLE_TOOLS"] = "true"

# Create conversation
conversation = Conversation(
    topic="Climate change solutions",
    agent_specs=[
        ("cynic", "engineer", "generalist"),
        ("dreamer", "marketing", "generalist")
    ]
)

# Run conversation
print(conversation.start_conversation())
for _ in range(4):
    print(conversation.next_turn())
```

## Testing

Run the test suite to verify tool calling functionality:

```bash
python test_tools.py
```

The test suite verifies:
- Tool creation and execution
- Model compatibility and parameter filtering
- Agent backward compatibility
- Tool integration with agents
- Error handling and fallback
- Web search tool functionality

### Example Usage

Run the example script to see tool calling in action:

```bash
python example_tool_usage.py
```

Make sure to set your `OPENAI_API_KEY` environment variable first.

## Performance Considerations

### Tool Usage Impact

- **Latency**: Tool calls add 2-5 seconds per response
- **Cost**: Web search is free (DuckDuckGo)
- **Rate Limits**: Built-in timeout and error handling
- **Caching**: Consider implementing result caching for repeated queries

### Optimization Tips

1. **Selective Usage**: Tools used only when relevant
2. **Timeout Configuration**: Adjust `TOOL_TIMEOUT` as needed
3. **Agent Types**: Limit tools for agents that don't need them
4. **Error Handling**: Robust fallback prevents conversation interruption

## Future Enhancements

### Planned Tools

- **Calculator**: Mathematical computations
- **Wikipedia**: Encyclopedia lookups
- **News API**: Current events
- **Code Execution**: Programming assistance

### Advanced Features

- **Tool Chaining**: Multiple tools in sequence
- **Custom Tools**: User-defined tool implementations
- **Tool Memory**: Remember tool results across turns
- **Tool Analytics**: Track tool usage patterns

## Troubleshooting

### Common Issues

1. **Tools Not Working**: Check `ENABLE_TOOLS=true`
2. **Import Errors**: Ensure `langchain-community` installed
3. **Timeout Errors**: Increase `TOOL_TIMEOUT` value
4. **Stop Parameter Errors**: Automatically handled by model compatibility checker
5. **Model Compatibility**: System automatically filters unsupported parameters

### Debug Mode

Enable verbose logging to debug tool usage:

```python
import logging
logging.getLogger("pandemonium.agents").setLevel(logging.DEBUG)
```

## Migration Guide

### From Existing Code

1. **No Changes Required**: Existing code works unchanged
2. **Enable Tools**: Set `ENABLE_TOOLS=true` to activate
3. **Test Gradually**: Enable tools for specific agent types first
4. **Monitor Performance**: Watch for increased response times

### Custom Agents

To add tool support to custom agents derived from MetaAgent:

```python
from pandemonium.agents.meta_agent import MetaAgent
from pandemonium.config import Config

class MyCustomAgent(MetaAgent):
    def __init__(self, temperament, expertise, trait="generalist"):
        # MetaAgent automatically handles tool integration
        super().__init__(temperament, expertise, trait)
        # Tools are automatically available when ENABLE_TOOLS=true
```

### Model Compatibility

The system automatically handles model compatibility for GPT-5 series:

```python
# No special configuration needed - system automatically:
# 1. Detects model capabilities
# 2. Filters unsupported parameters (like 'stop')
# 3. Provides graceful fallbacks
# 4. Prevents 400 errors from unsupported parameters
```

This implementation provides a robust, backward-compatible tool calling system that enhances Pandemonium agents with external capabilities while maintaining the existing conversational flow.
