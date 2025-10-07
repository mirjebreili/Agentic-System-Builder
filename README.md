# LangGraph Agent Planner

A minimal LangGraph application that performs planning only. The graph ingests a
user's first message (containing both the question and plugin documentation),
proposes Tree-of-Thought plan candidates, scores them, and pauses for human
approval before finalising a plan. No external tools are executed.

## Quick start

```bash
pip install -e . "langgraph-cli[inmem]"
langgraph dev
```

The graph returned by `langgraph dev` exposes the planner pipeline:

1. Parse the first user message into the natural-language question and per-plugin
   documentation.
2. Build a registry by merging the parsed docs with built-in metadata for the
   supported plugins (`HttpBasedAtlasReadByKey`,
   `membasedAtlasKeyStreamAggregator`).
3. Produce three ToT planning candidates, score them across coverage, I/O
   compatibility, simplicity, and constraint satisfaction, and compute
   softmax-based confidences.
4. Pause for HITL review. The workflow remains on the previously approved plan
   until the user replies with `APPROVE <index>`.

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

## Notes

- The planner operates in "plan-only" mode: tools are never executed.
- HITL responses support `APPROVE <index>` or `REVISE <instructions>`.
- Extend `src/tools/adapters/` and `src/tools/registry.py` to introduce new
  plugin metadata.
