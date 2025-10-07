# LangGraph Agent Planner

ToT Planner → HITL plan review (planner-only; no execution/scaffold/self-tests/report).

## Scope

This MVP **plans only**: it proposes an ordered plugin sequence and pauses at HITL for approval. It does **not** execute tools or scaffold projects.

## Quick start

```bash
# Install dependencies
 1. `pip install -e . "langgraph-cli[inmem]"`
 2. `cp .env.example .env` (edit MODEL/URL if needed)
 3. `langgraph dev --check`
 4. `langgraph dev` (the run pauses at HITL; planner-only)
```
 ## Scope
 This MVP **plans only**: it proposes an ordered plugin sequence and pauses at **HITL** for approval. It does **not** execute tools, scaffold projects, run self-tests, or generate reports.


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
