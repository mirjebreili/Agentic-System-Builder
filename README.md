# Agentic System Builder# Agentic System Builder (MVP)



An intelligent multi-agent system for automated task planning and execution using LangGraph. The system employs Tree-of-Thought planning with human-in-the-loop review to generate high-quality, validated execution plans.Pipeline: ToT Planner → HITL plan review → Agent self-tests → Deep Executor → Scaffold → Sandbox → Report.



## Features## Quickstart

1. `pip install -e . "langgraph-cli[inmem]"`

- **🎯 Goal Extraction**: Automatically extracts and clarifies high-level objectives2. `cp .env.example .env` (edit MODEL/URL if needed)

- **🔍 Plugin Analysis**: Discovers and analyzes available system tools and plugins3. `langgraph dev --check` to run a non-interactive smoke of the meta-graph.

- **🔗 Dependency Resolution**: Manages dependencies between system components4. Start a full run (`langgraph dev`) to approve the plan at the HITL pause and wait for scaffolding.

- **✂️ Task Decomposition**: Breaks complex tasks into manageable subtasks5. A generated project will appear in `projects/<slug>/`. Change into the directory and run `langgraph dev --check` followed by `pytest -q` to validate the scaffold before iterating.

- **🌳 Tree-of-Thought Planning**: Advanced planning with multiple reasoning paths

- **✅ Plan Validation**: Comprehensive validation including cycle detection and coverage analysisThe generated project follows LangGraph’s CLI app structure and is ready to extend with tools (MCP) later.

- **💯 Confidence Scoring**: Evaluates plan quality and success probability

- **👤 Human-in-the-Loop**: Interactive plan review and approval workflow### HITL plan review

- **📋 Plan Formatting**: Generates ordered, executable plan specifications

When the workflow pauses at the plan review step, resume it by sending one of

## Architecturethe following payloads:



The system implements a multi-stage workflow:- `{"action": "approve", "plan": {...}}`

- `{"action": "revise", "feedback": "..."}`

```

extract_goal → extract_system_elements → resolve_dependencies → For convenience the shorthand strings `"approve"` or `"revise"` are also

split_task → plan_tot → validate_plan → confidence → accepted.

review_plan → format_plan_order → END

```## Environment variables



### Key NodesThe `.env.example` file includes common configuration. Copy it to `.env` and override values as needed:



- **extract_goal**: Clarifies the user's objective- `LANGCHAIN_API_KEY` and `LANGCHAIN_ENDPOINT` configure access to LangChain services.

- **extract_system_elements**: Identifies available plugins and tools- Tracing is disabled by default with `LANGCHAIN_TRACING_V2=false` and `LANGSMITH_TRACING=false`; set them to `true` (and supply the API key/endpoint) to enable tracing.

- **resolve_dependencies**: Ensures proper component ordering
- **split_task**: Decomposes tasks into subtasks
- **plan_tot**: Generates detailed execution plan using Tree-of-Thought
- **validate_plan**: Validates completeness, detects cycles, checks tool availability
- **confidence**: Computes confidence score for the plan
- **review_plan**: Human approval checkpoint (workflow interrupts here)
- **format_plan_order**: Formats final execution order

### Validation Features

The plan validator ensures:
- All subtasks are addressed in the plan
- No circular dependencies exist
- All referenced tools/plugins are available
- Confidence thresholds are met
- Plans are non-empty and meaningful

## Installation

```bash
# Install dependencies
pip install -e .

# Install with LangGraph CLI
pip install -e . "langgraph-cli[inmem]"

# Copy environment configuration
cp .env.example .env
```

## Configuration

Edit `.env` file with your settings:

```bash
# LangChain/LangSmith Configuration
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGSMITH_TRACING=false

# Model Configuration
MODEL=gpt-4  # or your preferred model
```

## Usage

### Running the System

```bash
# Start the development server
langgraph dev

# Run a smoke test
langgraph dev --check
```

### Human-in-the-Loop Review

When the workflow pauses at the review step, approve or request changes:

**Approve the plan:**
```json
{"action": "approve"}
```

**Request revisions:**
```json
{"action": "revise", "feedback": "Please add error handling for API failures"}
```

Shorthand strings `"approve"` or `"revise"` are also accepted.

## Examples

See `examples/plugin_analyzer_demo.py` for a demonstration of the plugin extraction capabilities:

```bash
python examples/plugin_analyzer_demo.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run property-based tests
pytest tests/unit/test_property_based.py
```

## Project Structure

```
src/
├── agents/              # Core agent nodes
│   ├── graph.py        # Main workflow graph
│   ├── planner.py      # Tree-of-Thought planner
│   ├── plan_validator.py  # Plan validation
│   ├── goal_extractor.py  # Goal extraction
│   ├── dependency_resolver.py  # Dependency management
│   ├── plugin_analyzer.py  # Plugin/tool discovery
│   ├── splitter.py     # Task decomposition
│   ├── confidence.py   # Confidence scoring
│   ├── formatter.py    # Plan formatting
│   ├── hitl.py        # Human-in-the-loop review
│   ├── state.py       # State definition
│   └── state_validator.py  # State validation
├── config/             # Configuration management
├── llm/               # LLM client
├── prompts/           # Jinja2 prompt templates
└── utils/             # Logging, metrics, utilities

tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── meta/             # Repository structure tests
└── conftest.py       # Test fixtures

scripts/
└── bootstrap_tests.py  # Test suite generator

examples/
└── plugin_analyzer_demo.py  # Plugin extraction demo
```

## Development

The system uses:
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and message handling
- **Langfuse**: Observability and tracing
- **Pytest**: Testing framework
- **Hypothesis**: Property-based testing
- **Jinja2**: Prompt templating

### Key Components

- **State Management**: Centralized state with validation
- **Prompt Templates**: Jinja2 templates in `src/prompts/`
- **Logging**: Structured logging with context
- **Metrics**: Performance and quality metrics tracking
- **Error Handling**: Comprehensive error handling and retry logic

## Workflow Loops

The system includes intelligent retry loops:
- **Validation Loop**: Invalid plans automatically trigger replanning
- **Review Loop**: Human reviewers can request plan revisions
- **Interrupt Points**: Review step pauses for human approval

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
