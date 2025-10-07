# LangGraph Agent Planner

ToT Planner → HITL plan review (planner-only; no execution/scaffold/self-tests/report).

## Scope

This MVP **plans only**: it proposes an ordered plugin sequence and pauses at HITL for approval. It does **not** execute tools or scaffold projects.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
pip install "langgraph-cli[inmem]"

# Check that the graph is valid
# Note: the --check flag is illustrative; a successful server start confirms validity.
langgraph dev

# Run the planner, which will pause at the HITL step for user review.
langgraph dev
```

## First-message format

Supply the question and plugin documentation in the first user message. For
example:

```
سؤال: «مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده.»
پلاگین‌ها:
- partDeltaPluginHttpBasedAtlasReadByKey ...
- membasedAtlasKeyStreamAggregator ...
```

The planner reads the docs, constructs the registry, and proposes plans such as:

```
HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator
```