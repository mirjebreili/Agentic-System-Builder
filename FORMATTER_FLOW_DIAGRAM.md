# Formatter Node - Visual Flow Diagram

## Complete Workflow with Formatter

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENTIC SYSTEM WORKFLOW                      │
└─────────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                        ┌───────────────┐
                        │   plan_tot    │
                        │               │
                        │ • Generate K  │
                        │   alternative │
                        │   plans (ToT) │
                        │ • Validate    │
                        │ • Select best │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  confidence   │
                        │               │
                        │ • Compute     │
                        │   confidence  │
                        │   score       │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ review_plan   │◄──────────┐
                        │   (HITL)      │           │
                        │               │           │
                        │ • Interrupt   │           │
                        │ • Wait human  │           │
                        │ • Resume      │           │
                        └───────┬───────┘           │
                                │                   │
                                ▼                   │
                        ┌───────────────┐           │
                        │ route_after   │           │
                        │   _review     │           │
                        │               │           │
                        │ Decision:     │           │
                        │ replan?       │           │
                        └───────┬───────┘           │
                                │                   │
                        ┌───────┴───────┐           │
                        │               │           │
                  replan=True     replan=False      │
                        │               │           │
                        └───────────────┘           │
                                                    │
                                ▼                   │
                    ┌─────────────────────┐         │
                    │ format_plan_order   │         │
                    │    🆕 NEW NODE      │         │
                    │                     │         │
                    │ • Load prompts      │         │
                    │ • Prepare plan JSON │         │
                    │ • Invoke LLM        │         │
                    │ • Generate text     │         │
                    │ • Store in state    │         │
                    └──────────┬──────────┘         │
                               │                    │
                               ▼                    │
                              END                   │
                                                    │
                                                    │
└───────────────────────────────────────────────────┘


## Formatter Node Internal Flow

┌─────────────────────────────────────────────────────────────────────┐
│                      format_plan_order Function                      │
└─────────────────────────────────────────────────────────────────────┘

    Input: state (with approved plan)
       │
       ▼
    ┌─────────────────────┐
    │ Extract plan from   │
    │ state               │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Get LLM client      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Convert plan to     │
    │ JSON string         │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Load & render       │
    │ prompts:            │
    │ • format_system     │
    │ • format_user       │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Invoke LLM          │
    │                     │
    │ System: "You are an │
    │  expert technical   │
    │  writer..."         │
    │                     │
    │ User: "Format this  │
    │  plan: {...}"       │
    └──────────┬──────────┘
               │
               ▼
       ┌───────┴───────┐
       │               │
    Success?        Error?
       │               │
       ▼               ▼
    ┌─────┐      ┌─────────────┐
    │ LLM │      │ _fallback_  │
    │text │      │  format()   │
    └──┬──┘      └──────┬──────┘
       │                │
       └────────┬───────┘
                │
                ▼
    ┌─────────────────────┐
    │ Clean up markdown   │
    │ code fences         │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Add to messages     │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Store in scratch    │
    │ ['formatted_plan_   │
    │  order']            │
    └──────────┬──────────┘
               │
               ▼
    Output: Updated state


## Data Flow

┌──────────────┐
│   State In   │
│              │
│ • plan: {    │
│    goal,     │
│    nodes,    │
│    edges,    │
│    confidence│
│   }          │
│ • messages   │
│ • scratch    │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Formatter Node  │
│                  │
│  Uses:           │
│  • LLM client    │
│  • Prompts from  │
│    src/prompts/  │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  State Out   │
│              │
│ • plan       │ (unchanged)
│ • messages   │ (+ formatted text)
│ • scratch    │ (+ formatted_plan_order)
└──────────────┘


## Files Architecture

src/
├── agents/
│   ├── formatter.py        🆕 Formatter node implementation
│   ├── graph.py            ✏️  Updated with formatter
│   ├── planner.py          (existing)
│   ├── confidence.py       (existing)
│   └── hitl.py            (existing)
│
├── prompts/
│   ├── format_system.jinja 🆕 System prompt for formatter
│   ├── format_user.jinja   🆕 User prompt template
│   ├── plan_system.jinja   (existing)
│   └── plan_user.jinja     (existing)
│
└── llm/
    └── client.py           (existing - provides get_chat_model())


## Key Integration Points

1. **Import in graph.py**:
   from agents.formatter import format_plan_order

2. **Node registration**:
   g.add_node("format_plan_order", format_plan_order)

3. **Routing logic**:
   g.add_conditional_edges(
       "review_plan",
       route_after_review,
       {"plan_tot": "plan_tot", "format_plan_order": "format_plan_order"}
   )

4. **Final edge**:
   g.add_edge("format_plan_order", END)
```
