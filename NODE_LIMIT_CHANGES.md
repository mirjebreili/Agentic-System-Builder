# Node Limit Removal & Plugin Limit Implementation

## Overview
Removed the artificial limit on the number of nodes/steps in a plan and replaced it with a limit on the number of **different plugins** that can be used. This allows for more complex workflows while maintaining simplicity through plugin diversity constraints.

## Changes Made

### 1. Plan System Prompt (`src/prompts/plan_system.jinja`)

#### TASK Section Changes
**Before:**
```
- Build a tiny, linear plan (no branching unless absolutely necessary) that solves the user's goal.
- Prefer 2–3 steps. Never exceed 4 steps.
```

**After:**
```
- Build a linear plan (no branching unless absolutely necessary) that solves the user's goal.
- Use as many steps as needed to accomplish the goal correctly. No artificial limits on the number of nodes/steps.
- IMPORTANT: Use a maximum of 3 DIFFERENT plugins/tools. You can use the same plugin multiple times if needed, but limit the variety of plugins.
```

#### CONSTRAINTS Section Changes
**Before:**
```
- Minimize steps; do not invent missing capabilities.
```

**After:**
```
- Use a maximum of 3 DIFFERENT plugins/tools (you can reuse the same plugin multiple times).
- Do not invent missing capabilities.
```

#### SCORING HINTS Section Changes
**Before:**
```
- Simplicity: ≤3 nodes, zero branches.
```

**After:**
```
- Plugin Diversity: uses at most 3 different plugins/tools (reusing the same plugin is fine).
- Completeness: has all steps needed to accomplish the goal (don't artificially limit steps).
```

## Key Differences

### Old Approach (Node Limit)
- ❌ Maximum of 4 nodes/steps
- ❌ Preferred 2-3 steps
- ❌ Artificial constraint on workflow complexity
- ✅ Simple plans
- ❌ May not be sufficient for complex tasks

### New Approach (Plugin Limit)
- ✅ Unlimited number of nodes/steps
- ✅ Maximum of 3 different plugins
- ✅ Can reuse the same plugin multiple times
- ✅ Plans can be as complex as needed
- ✅ Maintains simplicity through plugin diversity constraint

## Examples

### Example 1: Same Plugin, Multiple Steps
**Valid Plan (uses only 2 different plugins, but 5 steps):**
```json
{
  "nodes": [
    {"id": "read_1", "tool": "httpBasedAtlasReadByKey", "reasoning": "Read first set of data"},
    {"id": "read_2", "tool": "httpBasedAtlasReadByKey", "reasoning": "Read second set of data"},
    {"id": "read_3", "tool": "httpBasedAtlasReadByKey", "reasoning": "Read third set of data"},
    {"id": "aggregate", "tool": "membasedAtlasKeyStreamAggregator", "reasoning": "Combine all data"},
    {"id": "aggregate_2", "tool": "membasedAtlasKeyStreamAggregator", "reasoning": "Final aggregation"}
  ]
}
```
✅ **Allowed**: Uses only 2 different plugins (httpBasedAtlasReadByKey and membasedAtlasKeyStreamAggregator) even though it has 5 steps.

### Example 2: Complex Multi-Step Workflow
**Valid Plan (3 plugins, 7 steps):**
```json
{
  "nodes": [
    {"id": "fetch_1", "tool": "PluginA"},
    {"id": "fetch_2", "tool": "PluginA"},
    {"id": "transform_1", "tool": "PluginB"},
    {"id": "transform_2", "tool": "PluginB"},
    {"id": "transform_3", "tool": "PluginB"},
    {"id": "output_1", "tool": "PluginC"},
    {"id": "output_2", "tool": "PluginC"}
  ]
}
```
✅ **Allowed**: Uses 3 different plugins with 7 total steps.

### Example 3: Too Many Different Plugins
**Invalid Plan (4 different plugins):**
```json
{
  "nodes": [
    {"id": "step1", "tool": "PluginA"},
    {"id": "step2", "tool": "PluginB"},
    {"id": "step3", "tool": "PluginC"},
    {"id": "step4", "tool": "PluginD"}
  ]
}
```
❌ **Not Allowed**: Uses 4 different plugins (exceeds limit of 3).

## Benefits

### 1. **More Flexibility**
- Plans can have as many steps as needed
- No artificial truncation of workflows
- Better coverage of complex requirements

### 2. **Maintains Simplicity**
- Limits complexity through plugin diversity
- Encourages reuse of familiar tools
- Easier to understand (fewer different tools to learn)

### 3. **Better for Iterative Tasks**
- Can loop through the same plugin multiple times
- Good for batch processing
- Allows multi-stage transformations with same tool

### 4. **Realistic Workflows**
- Matches real-world scenarios where you use few tools many times
- Example: Read from multiple sources → Aggregate → Filter → Format
- Example: Fetch → Parse → Fetch → Parse → Merge

## Use Cases

### Use Case 1: Multiple Data Sources
```
Task: Combine data from three different tracking code patterns

Plan:
1. httpBasedAtlasReadByKey (pattern: trackingCode_A*)
2. httpBasedAtlasReadByKey (pattern: trackingCode_B*)
3. httpBasedAtlasReadByKey (pattern: trackingCode_C*)
4. membasedAtlasKeyStreamAggregator (combine all)
5. membasedAtlasKeyStreamAggregator (final processing)

Plugins Used: 2 (within limit)
Steps: 5 (no limit!)
```

### Use Case 2: Multi-Stage Processing
```
Task: Process data through multiple transformations

Plan:
1. FetchPlugin (get data)
2. TransformPlugin (step 1)
3. TransformPlugin (step 2)
4. TransformPlugin (step 3)
5. TransformPlugin (step 4)
6. OutputPlugin (save results)

Plugins Used: 3 (within limit)
Steps: 6 (no limit!)
```

## Migration Notes

### For Existing Plans
- Plans with ≤4 steps will continue to work as before
- Plans can now expand beyond 4 steps if needed
- The system will favor plans that reuse plugins over plans that introduce new plugins

### For New Plans
- Focus on using maximum 3 different plugins
- Don't worry about the number of steps
- Reuse plugins for similar operations
- The LLM will automatically optimize for plugin diversity

## Testing

To test the changes:
1. Try a task that requires 5+ steps
2. Verify the plan is not artificially limited to 4 steps
3. Check that no more than 3 different plugins are used
4. Confirm plugins can be reused multiple times

## Summary

**Old Rule:** Maximum 4 nodes/steps  
**New Rule:** Maximum 3 different plugins (unlimited steps)

This change enables more complete and realistic workflows while maintaining simplicity through plugin diversity constraints! 🚀
