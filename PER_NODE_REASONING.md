# Per-Node Reasoning Feature - Implementation Summary

## Overview
Enhanced the planning system to include **reasoning for each individual node/step** in addition to overall plan reasoning. Now users can see why each specific step was selected in the workflow.

## Changes Made

### 1. PlanNode Model (`src/agents/planner.py`)
**Added `reasoning` field to PlanNode class:**
```python
class PlanNode(BaseModel):
    id: str
    prompt: str | None = None
    tool: str | None = None
    reasoning: str | None = None  # NEW FIELD - explains why this specific step is needed
```

### 2. Plan System Prompt (`src/prompts/plan_system.jinja`)
**Updated schema to request per-node reasoning:**
```json
{
  "nodes": [
    {
      "id": "<unique id>", 
      "tool": "<exact plugin/node name or null>", 
      "prompt": "<optional hint or null>",
      "reasoning": "<brief explanation of why this specific node/step is needed>"
    }
  ]
}
```

**Updated example to show node-level reasoning:**
- Each node now has its own reasoning explaining its role
- Overall plan still has reasoning for the approach
- LLM generates explanations for both levels

### 3. Format User Prompt (`src/prompts/format_user.jinja`)
**Updated format rules:**
- Display node reasoning immediately after each node
- Format: `X.Y Reason: <explanation>`
- Maintain overall plan reasoning
- Clear hierarchy: node reasoning → plan reasoning → confidence

### 4. Formatter Fallback (`src/agents/formatter.py`)
**Enhanced `_fallback_format()` function:**
- Extracts reasoning from each individual node
- Displays node reasoning with same numbering scheme
- Shows both per-node and per-plan reasoning
- Maintains clean formatting

## Output Format

### Complete Example:
```
1. [SELECTED]
   1.1 httpBasedAtlasReadByKey
   1.1 Reason: Fetches all keys matching trackingCode_* pattern with status field included
   1.2 filterActiveStatus
   1.2 Reason: Filters the results to only include items where status equals 'active'
   Reasoning: Two-step approach provides clear separation between data retrieval and filtering.
   Confidence: 0.92

2.
   2.1 httpBasedAtlasReadByKey
   2.1 Reason: Single query that combines pattern matching with status filtering
   Reasoning: More efficient single-step solution but less flexible for modifications.
   Confidence: 0.88

3.
   3.1 queryAtlas
   3.1 Reason: Retrieves all tracking codes from Atlas database
   3.2 parseResults
   3.2 Reason: Parses the JSON response into structured data format
   3.3 filterByStatus
   3.3 Reason: Applies status filter to extract only active items
   Reasoning: Comprehensive three-step approach with explicit parsing for better error handling.
   Confidence: 0.84
```

## Two-Level Reasoning Hierarchy

### Node-Level Reasoning
- **Purpose**: Explains what each specific step does
- **Scope**: Individual tool/node function
- **Example**: "Fetches data using pattern matching"
- **Format**: `X.Y Reason: <explanation>`

### Plan-Level Reasoning  
- **Purpose**: Explains why this overall approach was chosen
- **Scope**: Entire workflow strategy
- **Example**: "Two-step approach provides clear separation"
- **Format**: `Reasoning: <explanation>`

## Benefits

1. **Step-by-Step Clarity**: Users understand each individual step's purpose
2. **Learning**: Shows the role of each tool in the workflow
3. **Debugging**: Easy to identify which step might be causing issues
4. **Transparency**: Complete visibility into the decision-making process
5. **Comparison**: Can compare how different plans use similar tools differently
6. **Trust**: Builds confidence in each step of the execution

## Example for Your Query

**Query:** "از اطلس trackingCodeهایی رو بده که حالتشون active هست"

**Expected Output:**
```
1. [SELECTED]
   1.1 httpBasedAtlasReadByKey
   1.1 Reason: Uses pattern matching to find all trackingCode_* keys with included body data
   Reasoning: Direct single-step solution that efficiently retrieves active tracking codes.
   Confidence: 0.92

2.
   2.1 httpBasedAtlasReadByKey
   2.1 Reason: Fetches all tracking codes from Atlas database
   2.2 filterByActive
   2.2 Reason: Applies filtering logic to extract only active status items
   Reasoning: Two-step approach allows for more flexible filtering logic.
   Confidence: 0.88
```

## Testing

To verify per-node reasoning:
1. Run a query that generates multiple plan alternatives
2. Check that EACH node in EACH plan has its own reasoning
3. Verify the format shows node reasoning immediately after each step
4. Confirm overall plan reasoning is still displayed
5. Make sure reasoning helps differentiate between alternatives

## Data Flow

```
User Query
    ↓
LLM generates plans with:
    ├─ nodes (each with reasoning) ← NEW
    ├─ edges
    ├─ overall reasoning
    └─ confidence
    ↓
Formatter displays:
    ├─ Node 1
    ├─ Node 1 Reason ← NEW
    ├─ Node 2
    ├─ Node 2 Reason ← NEW
    ├─ Overall Reasoning
    └─ Confidence
```

Now you have **complete transparency** at both the plan level and the individual step level! 🎯
