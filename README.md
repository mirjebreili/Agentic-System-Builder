# Agentic System Builder (MVP)

Pipeline: ToT Planner → HITL plan review → Agent self-tests → Deep Executor → Scaffold → Sandbox → Report.

## Quickstart
1. `pip install -e . "langgraph-cli[inmem]"`
2. `cp .env.example .env` (edit MODEL/URL if needed)
3. `langgraph dev`  ← runs the meta-graph (Studio opens)
4. Start a run, approve the plan at the HITL pause, wait for scaffolding.
5. A generated project will appear in `projects/<slug>/`; `cd` into it and run `langgraph dev`.

The generated project follows LangGraph’s CLI app structure and is ready to extend with tools (MCP) later.

## Environment variables

The `.env.example` file includes common configuration. Copy it to `.env` and override values as needed:

- `LANGCHAIN_API_KEY` and `LANGCHAIN_ENDPOINT` configure access to LangChain services.
- Tracing is disabled by default with `LANGCHAIN_TRACING_V2=false` and `LANGSMITH_TRACING=false`; set them to `true` (and supply the API key/endpoint) to enable tracing.
