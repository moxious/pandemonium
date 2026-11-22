# Vision

`pandemonium` exists so that one person can put a hard question in front of a room of deliberately different minds and get back something better than a single model's single opinion.
It serves its author, the scripts and agents they point at it, and the author themselves when they take a seat in the room.
It turns a topic, a cast, and evaluation criteria into a transcript, a judged result, and, when asked, an artifact the room wrote together.
It owns exactly one thing: group deliberation among personas, from topic introduction through independent evaluation.

## Disagreement is the product

The value comes from friction between perspectives, not from any one agent being right.
Participants talk like people in a chatroom: short turns, one addressee at a time, a point of view they defend, and permission to disagree.
Chaos, cynicism, and contrarian traits are first-class casting choices, not bugs to be tuned out.
A room can co-author a deliverable such as a memo or a document, and the argument is how that deliverable gets made.
A human can join the room, and when they do they hold the floor: agents answer the human first and the broker summarizes for them.
Agents may use tools such as web search; when they do, evidence outranks assertion, and an agent concedes a point its sources contradict.
Changes that make agents more agreeable, more exhaustive, or more uniform in voice weaken the product and are resisted.

## Personas are data, and ways of looking live in Frame Shop

Curated temperaments and traits live in `personas.json`, not in code, and adding one never requires a code change.
Expertise comes from Frame Shop roles: a cast member is a temperament, a trait, and a role.
When Frame Shop is unavailable, `personas.json` expertise is the fallback.
The curated catalog is enumerable, so a caller can discover every valid value before casting a room.
A persona can also be generated at run time from a free-text description, and one that proves itself can be saved into the catalog.
Frame Shop, a separate library of frames and roles, is where reusable ways of looking at a problem and the stakes of a kind of job belong.
When a new persona is really a frame or a role, the contribution goes to Frame Shop and pandemonium consumes it from there, rather than growing a parallel catalog here.
Today the Frame Shop integration is optional; making it a required dependency is acceptable later.
Personas modeled on a real person are acceptable when there is enough source material to ground them, and they are not a current priority.
Shared chatroom behavior lives in one generic prompt; a persona describes only what makes it different.

## Facilitation, turns, and judgment are separate jobs

The broker is a participant who focuses the room against the criteria.
Who speaks next is always decided by a pluggable turn strategy, and new dynamics arrive as new strategies.
The default strategy is an LLM moderator that reads the room and names the next speaker; the decision still lives in the strategy, not in broker logic.
The judgment of a run comes from a fresh evaluator that did not take part in it, reads the whole conversation, and answers to the user's criteria.
The evaluator's result is structured data, and prose is a rendering of that data.
A run's conclusions can seed the introduction of a later run on the same subject, so follow-up rooms start from the last result.
Running the same room across providers or models and comparing the results is a supported use.

## The graph owns the conversation

Orchestration lives in one LangGraph state graph; application code builds the pieces and invokes it.
Conversation state is serializable and append-only; live agent objects and LLM clients travel in run config, never in state.
Agent context is bounded by design: a recent window, a pinned topic, and a broker summary for long runs.
Memory from earlier runs, whether a persona's or a room's, enters a turn through that same bounded context.
Any path that feeds an unbounded history to a model is a known defect, not an accepted cost.
Providers and models are a per-agent setting, so one room can mix vendors.
A run can be seeded so that casting, turn order, and broker rolls are reproducible up to model nondeterminism.

## Machine-first output

The CLI and an MCP server are both first-class ways for scripts and agent orchestrators to convene a room, over the same runner.
Structured output is the default: stdout carries only JSONL events, diagnostics go to stderr, and exit codes distinguish configuration errors from runtime failures.
Humans read a report rendered from that output rather than live console narration.
Every run can leave a structured transcript that records who spoke, when, and as which persona.
End-to-end tests exercise the full graph with mocked models, so the pipeline is verified without keys or network.

## Scope

Pandemonium is not a general multi-agent framework, not a chatbot, and not a frame library.
It runs locally for its author; a hosted, shared deployment is acceptable later and is not a current goal.
Topics, run output, and debate notes are the author's private material and stay out of the repository.
Dead code and superseded mechanisms are removed rather than kept behind flags.

A change aligns when it sharpens the friction between perspectives, widens who or what can sit in the room, adds personas as data or dynamics as strategies, sends ways of looking to Frame Shop, keeps state serializable and context bounded, or makes the run easier to drive and read from a machine.
A change should be resisted when it homogenizes agent voices, hard-codes personas or providers, decides turns outside a strategy, lets an evaluator take part in the run it judges, grows unbounded context, breaks the JSONL contract, duplicates in `personas.json` what belongs in Frame Shop, or turns the room into a general framework.
