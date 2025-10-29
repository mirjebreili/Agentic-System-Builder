# Test Suite Documentation

## Overview

This is a comprehensive test suite for the LangGraph agentic system builder. The tests are organized into multiple categories and use pytest as the test framework.

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── pytest.ini                      # Pytest configuration
├── meta/
│   └── test_repo_layout.py         # Project structure validation
├── unit/
│   ├── test_nodes.py               # Individual node tests
│   └── test_router.py              # Router logic tests
├── integration/
│   ├── test_graph_integration.py   # Full graph execution tests
│   ├── test_interrupts.py          # HITL interrupt/resume tests
│   ├── test_properties.py          # Property-based fuzzing tests
│   └── test_schemas.py             # SDK schema validation
└── cli/
    └── test_cli.py                 # CLI and SDK integration tests
```

## Quick Start

### 1. Bootstrap the Test Suite

```bash
# Basic setup (creates test files if missing)
python scripts/bootstrap_tests.py

# Force overwrite existing files
python scripts/bootstrap_tests.py --force

# Create sample app files if missing
python scripts/bootstrap_tests.py --with-samples

# Custom test prompts
python scripts/bootstrap_tests.py --prompts "build web app|analyze data|create plan"
```

### 2. Install Dependencies

```bash
pip install pytest pytest-asyncio hypothesis requests langgraph-sdk
```

### 3. Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run only fast tests (skip CLI and slow tests)
pytest -v -m "not cli and not slow"

# Run specific categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/cli/           # CLI tests only
pytest tests/meta/          # Meta tests only
```

## Test Categories

### Meta Tests (`tests/meta/`)

Validates the project structure and basic imports:
- ✅ Required files exist (langgraph.json, pyproject.toml, etc.)
- ✅ Can import graph and state modules
- ✅ Python version >= 3.11

**Run:** `pytest tests/meta/ -v`

### Unit Tests (`tests/unit/`)

Tests individual components in isolation:
- **test_nodes.py**: Tests each graph node with mocked dependencies
  - extract_system_elements
  - split_task
  - plan_tot (via confidence)
  - format_plan_order
- **test_router.py**: Tests conditional edge routing logic
  - Approval path
  - Replan path

**Run:** `pytest tests/unit/ -v`

### Integration Tests (`tests/integration/`)

Tests the full system with real components:
- **test_graph_integration.py**: End-to-end graph execution
  - Synchronous invoke
  - Asynchronous invoke
  - Streaming execution
  - Node verification
- **test_interrupts.py**: HITL interrupt and resume behavior
  - Interrupt at review_plan
  - Resume with approval/revision
- **test_properties.py**: Property-based fuzzing with Hypothesis
  - Random inputs should never crash
  - Always return valid state structure
- **test_schemas.py**: SDK schema validation
  - Assistant schemas exist
  - Minimal valid payloads work

**Run:** `pytest tests/integration/ -v`

### CLI Tests (`tests/cli/`)

Tests the LangGraph CLI and SDK:
- **test_cli.py**: Full CLI integration
  - `langgraph --help` works
  - `langgraph dev` server starts
  - SDK client operations (threads, runs)
  - Custom prompt parameterization
  - Streaming support

**Run:** `pytest tests/cli/ -v` (requires langgraph CLI installed)

## Fixtures

### Core Fixtures (conftest.py)

- **graph**: Compiled application graph with MemorySaver
- **fake_llm**: Deterministic fake LLM for testing
- **ok_tool**: Tool that always succeeds
- **bad_tool**: Tool that always fails
- **free_port**: Finds an available TCP port
- **dev_server**: Spawns and manages langgraph dev server
- **lg_client**: LangGraph SDK sync client
- **default_assistant_id**: Discovers assistant from server
- **cli_available**: Checks if langgraph CLI is installed

## Markers

Tests can be marked with pytest markers:

```python
@pytest.mark.cli        # Requires langgraph CLI
@pytest.mark.slow       # Long-running test
@pytest.mark.asyncio    # Async test
```

Skip marked tests:
```bash
pytest -v -m "not cli"      # Skip CLI tests
pytest -v -m "not slow"     # Skip slow tests
pytest -v -m "not cli and not slow"  # Skip both
```

## Environment Setup

The tests expect:
1. **Project Structure**:
   - `src/agents/graph.py` with exported `graph` object
   - `src/agents/state.py` with `AppState` definition
   - `langgraph.json` configuration file

2. **Dependencies**:
   - langgraph
   - langchain-core
   - pytest
   - pytest-asyncio
   - hypothesis (for property tests)
   - requests (for HTTP tests)
   - langgraph-sdk (for CLI tests)

3. **Environment Variables** (.env file):
   ```
   LLM_BASE_URL=http://your-llm-endpoint
   LLM_MODEL=your-model-name
   LLM_API_KEY=your-api-key
   TEMPERATURE=0.0
   ```

## Common Test Patterns

### Testing a Node

```python
def test_my_node():
    from src.agents.my_module import my_node
    
    state = {"messages": [], "goal": "test"}
    result = my_node(state)
    
    assert "expected_key" in result
    assert result["expected_key"] == "expected_value"
```

### Testing with Graph

```python
def test_with_graph(graph):
    from langchain_core.messages import HumanMessage
    
    initial_state = {
        "messages": [HumanMessage(content="Test prompt")]
    }
    
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "test-1"}}
    )
    
    assert "messages" in result
```

### Testing CLI with SDK

```python
@pytest.mark.cli
def test_with_sdk(lg_client, default_assistant_id):
    thread = lg_client.threads.create()
    
    run = lg_client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=default_assistant_id,
        input={"messages": [{"role": "user", "content": "Test"}]}
    )
    
    assert run["status"] in ["pending", "running", "interrupted"]
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio hypothesis requests
      - name: Run unit tests
        run: pytest tests/unit/ tests/integration/ -v -m "not cli"
      - name: Run coverage
        run: pytest --cov=src --cov-report=xml
```

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'graph' from 'src.agents.graph'`

**Solution**: Ensure your project structure matches the expected layout and the graph is properly exported.

### Server Timeout

**Problem**: `Server did not become ready within 30s`

**Solution**: 
- Check if port 2024 is already in use
- Verify langgraph CLI is installed: `langgraph --version`
- Check environment variables in .env

### Async Tests Not Running

**Problem**: Async tests are skipped

**Solution**: Install pytest-asyncio: `pip install pytest-asyncio`

### Property Tests Failing

**Problem**: Hypothesis tests fail with random inputs

**Solution**: This is expected! Fix the code to handle edge cases properly.

## Best Practices

1. **Keep tests fast**: Unit tests should run in < 1s each
2. **Use fixtures**: Don't duplicate setup code
3. **Mark slow tests**: Use `@pytest.mark.slow` for tests > 5s
4. **Test edge cases**: Empty inputs, None values, malformed data
5. **Use parametrize**: Test multiple inputs with `@pytest.mark.parametrize`
6. **Clean up resources**: Use fixtures with teardown for servers/files
7. **Descriptive names**: Test names should describe what they test
8. **One assertion**: Each test should verify one specific behavior

## Contributing

When adding new features:

1. Add unit tests for new nodes/functions
2. Add integration tests for new workflows
3. Update fixtures if new dependencies are needed
4. Mark tests appropriately (cli, slow, etc.)
5. Run full test suite before committing: `pytest -v`

## CI/CD Integration

The test suite is designed to work in CI environments:

- **Fast feedback**: Unit tests run in seconds
- **Selective testing**: Skip CLI tests in environments without the binary
- **Parallel execution**: Tests can run in parallel with `pytest -n auto`
- **Coverage reports**: Generate coverage for quality gates
- **Exit codes**: Non-zero exit on failure for CI pipelines

## Performance

Typical test execution times:

- **Meta tests**: < 1 second
- **Unit tests**: < 10 seconds
- **Integration tests**: < 60 seconds (without CLI)
- **CLI tests**: 1-3 minutes (includes server startup)
- **Full suite**: 3-5 minutes

Speed up tests:
```bash
# Skip slow tests
pytest -v -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```
