# Enhanced Workflow with Pattern Recognition

## Complete Flow Diagram

```mermaid
graph TD
    Start([User Input]) --> ExtractGoal[Extract Goal]
    
    ExtractGoal --> RecognizePattern[🆕 Recognize Plugin Pattern]
    
    RecognizePattern --> |Analyzes format| PatternOutput{Pattern Type?}
    
    PatternOutput --> |JSON_ARRAY<br/>JSON_OBJECT<br/>JSON_INLINE| JSONExtract[JSON Extraction]
    PatternOutput --> |MARKDOWN_LIST<br/>MARKDOWN_TABLE| RegexExtract[Regex Extraction]
    PatternOutput --> |PLAIN_TEXT| LLMExtract[LLM Semantic]
    PatternOutput --> |MIXED| HybridExtract[Hybrid Extraction]
    PatternOutput --> |NONE| AbstractPlan[Abstract Planning]
    
    JSONExtract --> ExtractElements[Extract System Elements]
    RegexExtract --> ExtractElements
    LLMExtract --> ExtractElements
    HybridExtract --> ExtractElements
    
    ExtractElements --> ResolveDeps[Resolve Dependencies]
    AbstractPlan --> SplitTask
    
    ResolveDeps --> SplitTask[Split Task]
    SplitTask --> PlanToT[Plan with ToT]
    PlanToT --> ValidatePlan[Validate Plan]
    
    ValidatePlan --> |Valid| Confidence[Compute Confidence]
    ValidatePlan --> |Needs Replan| PlanToT
    
    Confidence --> Review[HITL Review]
    Review --> |Approved| Format[Format Output]
    Review --> |Replan| PlanToT
    
    Format --> End([Final Plan])
    
    style RecognizePattern fill:#90EE90,stroke:#006400,stroke-width:3px
    style PatternOutput fill:#FFD700,stroke:#FF8C00,stroke-width:2px
    style ExtractElements fill:#87CEEB,stroke:#4682B4,stroke-width:2px
```

## Pattern Recognition Detail

```mermaid
graph LR
    Input[User Input] --> Analyze[LLM Pattern Analyzer]
    
    Analyze --> Detect1{Format Detection}
    Detect1 --> JSON[JSON Types]
    Detect1 --> MD[Markdown Types]
    Detect1 --> Text[Plain Text]
    Detect1 --> Mix[Mixed]
    
    JSON --> Struct1[Structure Analysis]
    MD --> Struct2[Structure Analysis]
    Text --> Struct3[Language Detection]
    Mix --> Struct4[Multi-Method Plan]
    
    Struct1 --> Strategy[Extraction Strategy]
    Struct2 --> Strategy
    Struct3 --> Strategy
    Struct4 --> Strategy
    
    Strategy --> Output[Pattern Metadata]
    Output --> Guide[Guides Next Node]
    
    style Analyze fill:#FF69B4,stroke:#C71585,stroke-width:2px
    style Output fill:#98FB98,stroke:#228B22,stroke-width:2px
```

## Before vs After

### Before (Old Flow)
```
[extract_goal] → [extract_system_elements] → [resolve_dependencies] → ...
                  ↓
                  Try LLM
                  ↓ (if fails)
                  Try JSON
                  ↓ (if fails)
                  Try Regex
                  ↓ (if fails)
                  Give up
```

### After (New Flow with Pattern Recognition)
```
[extract_goal] → [recognize_plugin_pattern] → [extract_system_elements] → ...
                         ↓                              ↓
                   Detects: JSON_ARRAY           Uses: json.loads()
                   Confidence: 0.95              Success in 1 try!
```

## Pattern Analysis Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Input                                               │
│    "[{"name":"plugin1"}, {"name":"plugin2"}]"              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. LLM Analyzes Input                                       │
│    - Checks syntax patterns                                 │
│    - Detects structure                                      │
│    - Identifies language                                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Pattern Metadata Generated                               │
│    {                                                        │
│      "format_type": "JSON_ARRAY",                          │
│      "structure_pattern": "flat_array",                    │
│      "language": "english",                                │
│      "extraction_strategy": "json_parse",                  │
│      "confidence": 0.95,                                   │
│      "has_plugins": true                                   │
│    }                                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Extraction Node Uses Pattern                            │
│    if pattern["format_type"] == "JSON_ARRAY":              │
│        plugins = json.loads(input)  # Direct parse         │
│    elif pattern["format_type"] == "PLAIN_TEXT":            │
│        plugins = llm_extract(input)  # Semantic            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Successful Extraction                                    │
│    plugins: [{"name": "plugin1"}, {"name": "plugin2"}]     │
│    system_elements: ["plugin1: ...", "plugin2: ..."]      │
└─────────────────────────────────────────────────────────────┘
```

## State Evolution

```
Initial State:
├── messages: [HumanMessage(...)]
└── goal: ""

After extract_goal:
├── messages: [...]
└── goal: "extracted goal text"

After recognize_plugin_pattern: 🆕
├── messages: [...]
├── goal: "..."
├── plugin_definition_pattern: {format, strategy, ...}
└── pattern_confidence: 0.95

After extract_system_elements:
├── messages: [...]
├── goal: "..."
├── plugin_definition_pattern: {...}
├── pattern_confidence: 0.95
├── plugins: [{name, goal, type}, ...]
├── system_elements: ["plugin1: ...", ...]
└── has_system_elements: true

After resolve_dependencies:
├── ...all above...
├── plugin_dependencies: {"A": ["B", "C"]}
└── dependency_order: ["B", "C", "A"]
```
