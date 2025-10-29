# Refactored Architecture Quick Reference

## Project Structure
```
src/
├── agents/
│   ├── graph.py              # Main LangGraph workflow
│   ├── state.py              # Simplified AppState definition
│   ├── plugin_analyzer.py    # Extract system elements from prompts
│   ├── splitter.py           # Split goals into atomic tasks
│   ├── planner.py            # Tree-of-Thought planning
│   ├── confidence.py         # Plan confidence scoring
│   ├── hitl.py               # Human-in-the-loop review
│   ├── formatter.py          # Format plan output
│   └── prompts_util.py       # Prompt directory helper
├── llm/
│   └── client.py             # LLM client wrapper
├── config/
│   └── settings.py           # Environment settings
└── utils/
    └── message_utils.py      # Message handling utilities
```

## Workflow Graph

```
extract_system_elements  →  split_task  →  plan_tot  →  confidence  →  review_plan
                                                                             ↓
                                                                         (approve?)
                                                                        ↓         ↓
                                                                   (yes)        (no)
                                                                        ↓         ↓
                                                              format_plan_order  ← (back to plan_tot)
                                                                        ↓
                                                                      END
```

## Node Descriptions

### 1. extract_system_elements
- **Purpose:** Extract plugins, APIs, components from user prompt
- **Input:** Initial user message
- **Output:** `system_elements`, `plugins`, `has_system_elements`

### 2. split_task
- **Purpose:** Break goal into atomic subtasks
- **Input:** User goal, system elements
- **Output:** `split_tasks` (list of subtask dictionaries)

### 3. plan_tot
- **Purpose:** Generate K=3 plan candidates using Tree-of-Thought
- **Input:** Goal, split tasks, system elements
- **Output:** `plan` (best candidate), `debug` (all candidates)

### 4. confidence
- **Purpose:** Calculate confidence score for the plan
- **Input:** Plan from planner
- **Output:** Updated `plan` with confidence score

### 5. review_plan (HITL)
- **Purpose:** Pause for human review/approval
- **Input:** Plan with confidence
- **Output:** `review` (action), `replan` (bool)
- **Resume with:** `{"action": "approve"}` or `{"action": "revise", "feedback": "..."}`

### 6. format_plan_order
- **Purpose:** Format plan execution order using LLM
- **Input:** Approved plan, all candidates
- **Output:** Formatted text in messages

## State Fields (AppState)

```python
{
    # Core
    "messages": List[AnyMessage],    # Conversation history
    "goal": str,                      # User's objective
    
    # Task decomposition
    "split_tasks": List[Dict],        # Atomic subtasks
    "system_elements": List[str],     # Extracted components
    "has_system_elements": bool,      # Detection flag
    
    # Plugin info
    "plugins": List[Dict],            # Available plugins
    
    # Planning
    "plan": Dict,                     # Current plan
    
    # HITL
    "review": Dict,                   # Review action/feedback
    "replan": bool,                   # Whether to replan
    
    # Debug
    "debug": Dict,                    # All plan candidates
}
```

## Running the System

### Local Development
```bash
langgraph dev
```

### Environment Variables Required
```
LLM_BASE_URL=http://your-llm-endpoint
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
TEMPERATURE=0.0
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

### Configuration File
`langgraph.json`:
```json
{
  "dependencies": ["./src"],
  "graphs": { "agent": "agents.graph:graph" },
  "env": "./.env"
}
```

## Key Simplifications

1. **State:** Reduced from 13 to 9 essential fields
2. **Confidence:** Simplified from 4-factor to 2-factor scoring
3. **Plugin Analyzer:** Consolidated extraction logic
4. **Removed:** fileops.py, executor imports, debug prints
5. **Cleaner:** Template rendering, error handling

## Testing

```python
# Test graph loading
from src.agents.graph import graph
print(f"Nodes: {list(graph.nodes.keys())}")

# Compile check
python -m py_compile src/agents/*.py
```

## Maintenance Notes

- All nodes are pure functions that take/return state dicts
- No side effects except LLM calls
- Logging uses Python's standard logging module
- LangGraph handles persistence automatically
- HITL interrupts work with `interrupt_before=["review_plan"]`
