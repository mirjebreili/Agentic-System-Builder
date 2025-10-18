# Adding Reasoning to Plans - Implementation Summary

## Overview
Enhanced the planning and formatting system to include **reasoning** for why each plan was selected. Now each alternative plan shows its approach and rationale.

## Changes Made

### 1. Plan Model (`src/agents/planner.py`)
**Added `reasoning` field to the Plan class:**
```python
class Plan(BaseModel):
    goal: str
    nodes: list[PlanNode]
    edges: list[PlanEdge]
    confidence: float | None = None
    reasoning: str | None = None  # NEW FIELD
```

### 2. Plan System Prompt (`src/prompts/plan_system.jinja`)
**Updated schema to request reasoning:**
- Added `"reasoning"` field to the plan schema
- Updated example to show reasoning in action
- LLM now generates brief explanations for each plan alternative

**Example reasoning added:**
```json
{
  "reasoning": "This plan directly reads the required keys using pattern matching, then aggregates them efficiently in a single step. Simple and focused on the exact goal."
}
```

### 3. Format User Prompt (`src/prompts/format_user.jinja`)
**Updated instructions to display reasoning:**
- Request display of reasoning for each plan
- Added reasoning to the example format
- Format: `Reasoning: <explanation>`

### 4. Formatter Fallback (`src/agents/formatter.py`)
**Enhanced `_fallback_format()` function:**
- Extracts reasoning from each plan
- Displays reasoning before confidence score
- Only shows reasoning if available (optional field)

## Output Format

### Before (without reasoning):
```
1. [SELECTED]
   1.1 httpBasedAtlasReadByKey
   1.2 filterResults
   Confidence: 0.92

2.
   2.1 httpBasedAtlasReadByKey
   Confidence: 0.88
```

### After (with reasoning):
```
1. [SELECTED]
   1.1 httpBasedAtlasReadByKey
   1.2 filterResults
   Reasoning: Direct approach using pattern matching followed by filtering. Efficient two-step solution.
   Confidence: 0.92

2.
   2.1 httpBasedAtlasReadByKey
   Reasoning: Single-step solution that relies on advanced query parameters. Simpler but less flexible.
   Confidence: 0.88

3.
   3.1 getAllData
   3.2 filterByPattern
   3.3 extractActive
   Reasoning: Comprehensive approach that fetches all data first, then applies multiple filters. More robust but slower.
   Confidence: 0.84
```

## Benefits

1. **Transparency**: Users understand why each plan was generated
2. **Decision Making**: Clear rationale for plan selection
3. **Learning**: Users can see different approaches to the same problem
4. **Trust**: Builds confidence in the AI's decision-making process
5. **Debugging**: Easier to understand if a plan doesn't work as expected

## How It Works

```
User Query
    ↓
plan_tot generates K=3 plans
    ↓ Each plan now includes:
    ├─ nodes (steps)
    ├─ edges (flow)
    ├─ confidence score
    └─ reasoning ← NEW
    ↓
Best plan selected
    ↓
formatter retrieves all plans
    ↓
Displays all alternatives with:
    ├─ Numbered modules
    ├─ Reasoning ← NEW
    └─ Confidence
```

## Testing

To see the reasoning in action:
1. Run a query that triggers `plan_tot`
2. The output will now show reasoning for each plan alternative
3. Verify that:
   - Each plan has a meaningful reasoning explanation
   - The selected plan is clearly marked
   - Reasoning helps explain the differences between alternatives

## Example Use Case

**Query:** "از اطلس trackingCodeهایی رو بده که حالتشون active هست"

**Output will show:**
- Plan 1: Why it uses pattern matching + filter
- Plan 2: Why it uses single-step approach
- Plan 3: Why it uses comprehensive fetching

Each with its reasoning explaining the tradeoffs!
