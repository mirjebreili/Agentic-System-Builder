# Plan Order Formatter Node

## Overview
A new LLM-powered node `format_plan_order` has been added to the agent system that formats the approved plan into a readable text description showing the execution order.

## Files Created/Modified

### 1. New File: `src/agents/formatter.py`
This is a separate script containing the LLM-based formatter node implementation.

**Key Features:**
- `format_plan_order(state)`: Main function that uses an LLM to format the plan
- `_fallback_format(plan)`: Fallback formatter if LLM fails
- Uses LangChain's LLM client with proper system and user prompts
- Creates a well-formatted, professional text output showing:
  - Plan goal and confidence score
  - Total number of steps
  - Execution sequence with step numbers
  - Tool or prompt information for each step
  - Conditional execution paths (if any)
  - Next steps for each node
- Error handling with automatic fallback to simple formatting

**LLM Integration:**
- Loads prompts from `src/prompts/format_system.jinja` and `src/prompts/format_user.jinja`
- Sends the entire plan as JSON to the LLM
- LLM analyzes the graph structure and generates human-readable documentation
- Cleans up markdown artifacts from LLM response

### 2. New File: `src/prompts/format_system.jinja`
System prompt that defines the LLM's role as a technical writer and workflow documentation specialist.

**Responsibilities Defined:**
- Analyze plan structure (nodes and edges)
- Determine correct execution order
- Create professional documentation with clear formatting
- Handle conditional logic and branching
- Make output accessible to both technical and non-technical readers

**Formatting Guidelines:**
- Use ASCII art borders and separators
- Professional tone and structure
- Clear section headers
- Proper indentation and numbering

### 3. New File: `src/prompts/format_user.jinja`
User prompt template that provides the plan data and specific formatting instructions.

**Contents:**
- Plan JSON data injection point (`{{ plan_json }}`)
- Detailed formatting instructions
- Output format specifications
- Step-by-step requirements for the LLM

### 4. Modified File: `src/agents/graph.py`
Updated the graph to include the new formatter node in the workflow.

**Changes:**
1. **Import added**: `from agents.formatter import format_plan_order`
2. **Route updated**: `route_after_review()` now routes to `format_plan_order` instead of `END` when approved
3. **Node added**: `g.add_node("format_plan_order", format_plan_order)`
4. **Edges updated**:
   - Conditional edge from `review_plan` to either `plan_tot` (if replan) or `format_plan_order` (if approved)
   - Direct edge from `format_plan_order` to `END`

## Workflow Order

The updated workflow now follows this sequence:

1. **plan_tot** → Generate plans using Tree of Thought
2. **confidence** → Compute plan confidence
3. **review_plan** → Human-in-the-loop review (interrupt point)
4. **route_after_review** → Decision point:
   - If `replan=True` → go back to **plan_tot**
   - If `replan=False` → continue to **format_plan_order**
5. **format_plan_order** → LLM formats the approved plan (NEW NODE)
6. **END** → Workflow complete

## LLM Output Format

The LLM generates output similar to:

```
============================================================
PLAN EXECUTION ORDER
============================================================

Goal: <goal description>
Confidence: <confidence score>

Total Steps: <number>

------------------------------------------------------------
EXECUTION SEQUENCE:
------------------------------------------------------------

Step 1: <node_id>
  Type: Tool Execution / Prompt Execution
  Tool: <tool_name> / Prompt: <prompt preview>
  Condition: <condition if any>
  Next: <next_node_ids>

...

============================================================
END OF PLAN
============================================================
```

## State Management

The formatter stores data in two places:
- **messages**: Adds a new assistant message with the formatted plan text
- **scratch['formatted_plan_order']**: Stores the formatted text for easy programmatic access

## Benefits

1. **LLM-Powered**: Uses AI to create natural, readable documentation
2. **Flexible**: LLM can adapt to different plan structures and complexities
3. **Separate Script**: The formatter logic is isolated in its own module for maintainability
4. **Prompt-Based**: Easy to modify behavior by editing prompt templates
5. **Clear Output**: Provides a human-readable summary of the execution plan
6. **Post-Approval**: Only runs after the plan is approved, avoiding unnecessary formatting
7. **Graph-Aware**: LLM understands and explains the node/edge structure
8. **Error Resilient**: Includes fallback formatting if LLM fails
9. **Logging**: Includes proper logging for debugging and monitoring
10. **Professional**: Creates documentation suitable for technical and business audiences

## Usage

The formatter automatically runs after plan approval. The formatted output will appear in:
- The messages list as an assistant message
- The scratch pad under the key `formatted_plan_order`

You can access it programmatically:
```python
formatted_text = state.get("scratch", {}).get("formatted_plan_order", "")
```

## Customization

To customize the formatting behavior:
1. Edit `src/prompts/format_system.jinja` to change the LLM's role and responsibilities
2. Edit `src/prompts/format_user.jinja` to modify instructions and output format
3. Modify `_fallback_format()` in `formatter.py` to change the fallback behavior
