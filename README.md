# Agentic System Builder (MVP)

Pipeline: ToT Planner → HITL plan review → Agent self-tests → Deep Executor → Scaffold → Sandbox → Report.

## Quickstart
1. `pip install -e . "langgraph-cli[inmem]"`
2. `cp .env.example .env` (edit MODEL/URL if needed)
3. `langgraph dev --check` to run a non-interactive smoke of the meta-graph.
4. Start a full run (`langgraph dev`) to approve the plan at the HITL pause and wait for scaffolding.
5. A generated project will appear in `projects/<slug>/`. Change into the directory and run `langgraph dev --check` followed by `pytest -q` to validate the scaffold before iterating.

The generated project follows LangGraph’s CLI app structure and is ready to extend with tools (MCP) later.

### HITL plan review

When the workflow pauses at the plan review step, resume it by sending one of
the following payloads:

- `{"action": "approve", "plan": {...}}`
- `{"action": "revise", "feedback": "..."}`

For convenience the shorthand strings `"approve"` or `"revise"` are also
accepted.

## Environment variables

The `.env.example` file includes common configuration. Copy it to `.env` and override values as needed:

- `LANGCHAIN_API_KEY` and `LANGCHAIN_ENDPOINT` configure access to LangChain services.
- Tracing is disabled by default with `LANGCHAIN_TRACING_V2=false` and `LANGSMITH_TRACING=false`; set them to `true` (and supply the API key/endpoint) to enable tracing.

## Supplying message attachments

The compiled LangGraph automatically inspects the incoming state for common attachment fields—`attachments`, `files`, `input_files`, or `uploaded_files`. Any entries found in those collections are normalized into LangChain-style `{"type": "file", ...}` blocks and appended to the latest user message before the workflow starts. Text-based attachments are also folded into the `input_text` string so downstream nodes see the full context.

You can pass attachments as dictionaries, filesystem paths, or raw bytes. The helper will ignore non-textual payloads (for example images) when building the `input_text` fallback, but they are still added to the message content for tool usage. A minimal example when invoking the compiled graph directly looks like this:

```python
from asb.agent.graph import make_graph

graph = make_graph()

state = {
    "input_text": "Summarise the requirements in the attached docs.",
    "attachments": [
        {"type": "file", "file_path": "./docs/requirements.txt"},
        {"type": "file", "data": b"Title: Notes\nDetails...", "mime_type": "text/plain"},
    ],
}

result = graph.invoke(state)
```

The same structure works when driving the agent through `langgraph dev` (or the LangGraph API): include the attachment payloads in the request body alongside your normal `input_text`/`messages`. No additional configuration is required.
