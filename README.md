# LangGraph Agent Planner

This is a minimal scaffold for a LangGraph agent that can be run with `langgraph dev`.

## Quick start

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the development server:**
    ```bash
    langgraph dev
    ```

4.  **Send a message to the graph:**
    In a separate terminal, send a POST request to the running server. The input should be a JSON object with an `input` key, which contains another JSON object with an `initial_message` key.

    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"input": {"initial_message": "hello world"}}' \
    http://127.0.0.1:8000/planner/invoke
    ```

    You should receive a response similar to this:

    ```json
    {
      "output": {
        "candidates": [
          {
            "plan": [
              "HttpBasedAtlasReadByKey",
              "membasedAtlasKeyStreamAggregator"
            ],
            "rationale": "placeholder"
          }
        ],
        "chosen": 0,
        "note": "stub pipeline — logic to be implemented in Task 2+"
      }
    }
    ```