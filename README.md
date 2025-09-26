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
