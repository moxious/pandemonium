"""
LangGraph conversation orchestrator for Pandemonium.

This module defines the conversation graph that drives the multi-agent discussion.
The graph handles topic introduction, speaker selection, agent responses,
round tracking, and final evaluation.
"""

import operator
import logging
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pandemonium.config import Config

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State schema for the LangGraph conversation graph."""
    messages: Annotated[List[BaseMessage], operator.add]  # append-only via reducer
    topic: str
    round_count: int
    max_rounds: int
    current_speaker: str
    speakers_this_round: List[str]
    planned_round_order: List[str]
    evaluation_criteria: str
    round_boundaries: List[int]  # message index where each round starts
    summary: str  # running summary for long conversations
    # Message-based strategy fields
    message_count: int  # total agent/broker turns produced
    max_messages: int  # conversation length limit (0 = round-based mode)
    speaker_history: Annotated[List[str], operator.add]  # append-only speaker sequence


def introduce_topic(state: ConversationState, config: RunnableConfig) -> dict:
    """Introduce the conversation topic and plan the first round."""
    configurable = config.get("configurable", {})
    broker = configurable["broker"]
    agents = configurable["agents"]
    turn_strategy = configurable["turn_strategy"]
    transcript_logger = configurable.get("transcript_logger")

    agent_names = list(agents.keys())
    introduction = broker.introduce_topic(state["topic"])

    # Plan first round
    planned_order = turn_strategy.plan_round(agent_names, broker.name, 0)
    if configurable.get("broker_mode") == "silent":
        planned_order = [s for s in planned_order if s != broker.name]

    if transcript_logger:
        transcript_logger.log_entry(
            round_number=0,
            speaker=broker.name,
            speaker_type="broker",
            content=introduction,
        )

    return {
        "messages": [HumanMessage(content=introduction)],
        "planned_round_order": planned_order,
        "speakers_this_round": [],
        "round_count": 0,
        "round_boundaries": [0],  # round 0 starts at message index 0
        "message_count": 0,
        "speaker_history": [],
    }


def select_speaker(state: ConversationState, config: RunnableConfig) -> dict:
    """Select the next speaker."""
    configurable = config.get("configurable", {})
    turn_strategy = configurable["turn_strategy"]

    if turn_strategy.is_message_based:
        agents = configurable["agents"]
        broker = configurable["broker"]
        agent_names = list(agents.keys())
        broker_mode = configurable.get("broker_mode", "silent")

        agent_metadata = {}
        for name, agent in agents.items():
            agent_metadata[name] = {
                "temperament": getattr(agent, "temperament", ""),
                "expertise": getattr(agent, "expertise", ""),
            }

        speaker = turn_strategy.select_next_speaker(
            agent_names=agent_names,
            broker_name=broker.name if broker_mode == "active" else None,
            messages=state["messages"],
            speaker_history=state.get("speaker_history", []),
            agent_metadata=agent_metadata,
            message_count=state.get("message_count", 0),
            max_messages=state.get("max_messages", 30),
        )
    else:
        speakers_this_round = state["speakers_this_round"]
        planned_order = state["planned_round_order"]

        next_index = len(speakers_this_round)
        if next_index < len(planned_order):
            speaker = planned_order[next_index]
        else:
            speaker = planned_order[0]

    logger.debug(f"Selected speaker: {speaker}")

    return {
        "messages": [],
        "current_speaker": speaker,
    }


def agent_respond(state: ConversationState, config: RunnableConfig) -> dict:
    """Have the current speaker generate a response."""
    configurable = config.get("configurable", {})
    agents = configurable["agents"]
    broker = configurable["broker"]
    turn_strategy = configurable["turn_strategy"]
    transcript_logger = configurable.get("transcript_logger")

    speaker_name = state["current_speaker"]

    # Look up the agent (could be a regular agent or the broker)
    if speaker_name == broker.name:
        agent = broker
        speaker_type = "broker"
    else:
        agent = agents.get(speaker_name)
        speaker_type = "agent"
        if agent is None:
            logger.error(f"Unknown speaker: {speaker_name}")
            return {"messages": [], "speakers_this_round": state["speakers_this_round"]}

    # Build context: pinned intro + optional summary + recent messages/rounds
    all_messages = state["messages"]
    intro_message = all_messages[0] if all_messages else None

    if turn_strategy.is_message_based:
        # Message-based: sliding window of last N messages
        window_size = Config.CONTEXT_MESSAGES
        if len(all_messages) > window_size:
            start_idx = len(all_messages) - window_size
        else:
            start_idx = 0
    else:
        # Round-based: window by round boundaries
        round_boundaries = state["round_boundaries"]
        current_round = state["round_count"]
        context_rounds = Config.CONTEXT_ROUNDS
        earliest_round = max(0, current_round - context_rounds + 1)
        if earliest_round < len(round_boundaries):
            start_idx = round_boundaries[earliest_round]
        else:
            start_idx = 0

    # Assemble context for the agent
    context_messages = []
    if intro_message and start_idx > 0:
        # Intro fell outside the window — pin it at the top
        context_messages.append(intro_message)
    summary = state.get("summary", "")
    if summary:
        context_messages.append(HumanMessage(content=f"[Summary of earlier discussion: {summary}]"))
    context_messages.extend(all_messages[start_idx:])

    # Generate response
    msg_num = state.get("message_count", 0) if turn_strategy.is_message_based else state["round_count"]
    label = "Msg" if turn_strategy.is_message_based else "Round"
    logger.info(f"{label} {msg_num}: {speaker_name} responding")
    result = agent.respond(state["topic"], context_messages=context_messages)
    response = result["content"]

    # Track token usage
    token_tracker = configurable.get("token_tracker")
    if token_tracker:
        token_tracker.track(result["input_tokens"], result["output_tokens"])

    new_speakers = state["speakers_this_round"] + [speaker_name]

    if transcript_logger:
        persona_config = None
        if hasattr(agent, 'temperament') and hasattr(agent, 'expertise'):
            persona_config = {
                "temperament": agent.temperament,
                "expertise": agent.expertise,
            }
        transcript_logger.log_entry(
            round_number=state["round_count"],
            speaker=speaker_name,
            speaker_type=speaker_type,
            content=response,
            persona_config=persona_config,
            token_count=result["input_tokens"] + result["output_tokens"],
        )

    return {
        "messages": [AIMessage(content=f"{speaker_name}: {response}")],
        "speakers_this_round": new_speakers,
        "message_count": state.get("message_count", 0) + 1,
        "speaker_history": [speaker_name],
    }


def check_round(state: ConversationState, config: RunnableConfig) -> dict:
    """Check if the round is complete and plan the next one if needed."""
    configurable = config.get("configurable", {})
    turn_strategy = configurable["turn_strategy"]
    agents = configurable["agents"]
    broker = configurable["broker"]

    token_tracker = configurable.get("token_tracker")

    if turn_strategy.is_message_based:
        # Message-based: no round management. Only generate summary periodically.
        summary = state.get("summary", "")
        message_count = state.get("message_count", 0)
        if message_count > 0 and message_count % Config.SUMMARY_AFTER_MESSAGES == 0:
            summary = _generate_summary(state, broker, token_tracker)
        return {"messages": [], "summary": summary}

    # Round-based path (unchanged)
    speakers_this_round = state["speakers_this_round"]
    planned_order = state["planned_round_order"]

    if turn_strategy.is_round_complete(speakers_this_round, planned_order):
        new_round = state["round_count"] + 1
        logger.info(f"Round {state['round_count']} complete. Starting round {new_round}.")

        # Plan next round
        agent_names = list(agents.keys())
        new_planned = turn_strategy.plan_round(agent_names, broker.name, new_round)
        if configurable.get("broker_mode") == "silent":
            new_planned = [s for s in new_planned if s != broker.name]

        # Record where the new round starts in the message list
        new_boundaries = state["round_boundaries"] + [len(state["messages"])]

        # Generate a running summary once the conversation is long enough
        summary = state.get("summary", "")
        if new_round >= Config.SUMMARY_AFTER_ROUNDS:
            summary = _generate_summary(state, broker, token_tracker)

        return {
            "messages": [],
            "round_count": new_round,
            "speakers_this_round": [],
            "planned_round_order": new_planned,
            "round_boundaries": new_boundaries,
            "summary": summary,
        }

    # Round not complete yet, continue
    return {"messages": []}


def _generate_summary(state: ConversationState, broker, token_tracker=None) -> str:
    """Use the broker's LLM to generate a brief recap of the conversation so far."""
    all_messages = state["messages"]
    history = "\n".join(msg.content for msg in all_messages)
    prompt = [
        SystemMessage(content="You are a neutral summarizer. Produce a concise 2-3 sentence recap of the key points and positions in this discussion so far. Do not add opinions."),
        HumanMessage(content=history),
    ]
    try:
        response = broker.llm.invoke(prompt)
        if token_tracker:
            usage = getattr(response, 'usage_metadata', None) or {}
            token_tracker.track(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        logger.debug(f"Generated conversation summary: {response.content[:100]}...")
        return response.content
    except Exception as e:
        logger.warning(f"Failed to generate summary: {e}")
        return state.get("summary", "")


def should_continue(state: ConversationState) -> str:
    """Determine if the conversation should continue or conclude."""
    # Message-based termination
    max_messages = state.get("max_messages", 0)
    if max_messages > 0:
        if state.get("message_count", 0) >= max_messages:
            return "evaluate"
        return "next_speaker"

    # Round-based termination
    if state["round_count"] >= state["max_rounds"]:
        return "evaluate"

    return "next_speaker"


def evaluate(state: ConversationState, config: RunnableConfig) -> dict:
    """Create an evaluator agent and conclude the conversation."""
    from pandemonium.agents.evaluator_agent import EvaluatorAgent

    configurable = config.get("configurable", {})
    transcript_logger = configurable.get("transcript_logger")

    topic = state["topic"]
    evaluation_criteria = state.get("evaluation_criteria", "pick most interesting or important issues")

    evaluation_prompt = f"""
You are an independent evaluator reviewing a complete chatroom discussion about "{topic}".

You have access to the entire conversation history.

First, you will read your evaluation criteria, these frame how you understand the
conversation.  Your evaluation criteria are: {evaluation_criteria}

Next, you will read through the entire chatroom discussion.

Next, you will synthesize the best outcome of the conversation, based on the evaluation criteria.

Finally, you will provide a brief summary of the best parts of the conversation, and end with a
statement that says:

Result: (your synthesis of the best outcome in no more than 3 sentences)
"""

    evaluator = EvaluatorAgent(evaluation_prompt, model='gpt-5', temperature=0.5)

    # Format all messages for evaluation
    history_parts = [msg.content for msg in state["messages"]]
    conversation_history = "\n".join(history_parts)

    result = evaluator.evaluate_conversation(conversation_history)
    evaluator_summary = result["content"]

    # Track token usage
    token_tracker = configurable.get("token_tracker")
    if token_tracker:
        token_tracker.track(result["input_tokens"], result["output_tokens"])

    max_messages = state.get("max_messages", 0)
    if max_messages > 0:
        length_desc = f"{state.get('message_count', 0)} messages"
    else:
        length_desc = f"{state['max_rounds']} rounds"

    conclusion = f"""\n--- Conversation Complete ---

We've completed {length_desc} of discussion on "{topic}".
Thank you to all participants for sharing their unique perspectives!

Final evaluation by independent assessor:
{evaluator.name}: {evaluator_summary}"""

    if transcript_logger:
        transcript_logger.log_entry(
            round_number=state["round_count"],
            speaker=evaluator.name,
            speaker_type="evaluator",
            content=evaluator_summary,
            token_count=result["input_tokens"] + result["output_tokens"],
        )

    return {
        "messages": [HumanMessage(content=conclusion)],
    }


def build_conversation_graph() -> StateGraph:
    """Build and compile the conversation orchestration graph."""
    graph = StateGraph(ConversationState)

    graph.add_node("introduce_topic", introduce_topic)
    graph.add_node("select_speaker", select_speaker)
    graph.add_node("agent_respond", agent_respond)
    graph.add_node("check_round", check_round)
    graph.add_node("evaluate", evaluate)

    graph.set_entry_point("introduce_topic")
    graph.add_edge("introduce_topic", "select_speaker")
    graph.add_edge("select_speaker", "agent_respond")
    graph.add_edge("agent_respond", "check_round")

    graph.add_conditional_edges(
        "check_round",
        should_continue,
        {
            "next_speaker": "select_speaker",
            "evaluate": "evaluate",
        }
    )

    graph.add_edge("evaluate", END)

    return graph.compile()
