import os
import csv
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langsmith import Client
from typing import Optional, List, Dict, Any

def load_langsmith_config() -> Dict[str, str]:
    """Load LangSmith configuration from environment variables."""
    # Load environment variables from .env file
    load_dotenv()
    
    config = {
        'tracing': os.getenv('LANGSMITH_TRACING', 'false'),
        'endpoint': os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com'),
        'api_key': os.getenv('LANGSMITH_API_KEY'),
        'project': os.getenv('LANGSMITH_PROJECT')
    }
    
    # Validate required configuration
    if not config['api_key']:
        raise ValueError("LANGSMITH_API_KEY environment variable is required")
    if not config['project']:
        raise ValueError("LANGSMITH_PROJECT environment variable is required")
    
    return config

def download_langsmith_traces(
    project_name: str,
    api_key: str,
    endpoint: str = "https://api.smith.langchain.com",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    run_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Download traces from LangSmith.
    
    Args:
        project_name: Name of the LangSmith project
        api_key: LangSmith API key
        endpoint: LangSmith API endpoint
        start_time: Start time for filtering runs
        end_time: End time for filtering runs
        run_type: Filter by run type (e.g., "llm", "chain", "tool")
        limit: Maximum number of runs to retrieve
    
    Returns:
        List of run data dictionaries
    """
    # Initialize LangSmith client
    client = Client(
        api_url=endpoint,
        api_key=api_key
    )
    
    # Set up query parameters
    query_params = {
        'project_name': project_name
    }
    
    if start_time:
        query_params['start_time'] = start_time
    if end_time:
        query_params['end_time'] = end_time
    if run_type:
        query_params['run_type'] = run_type
    if limit:
        query_params['limit'] = limit
    
    # Retrieve runs
    print(f"Downloading traces from project: {project_name}")
    runs = list(client.list_runs(**query_params))
    
    print(f"Retrieved {len(runs)} runs")
    
    # Convert runs to serializable format
    runs_data = []
    for run in runs:
        # Only include attributes that actually exist on the Run object
        run_dict = {
            'id': str(run.id),
            'name': run.name,
            'run_type': run.run_type,
            'start_time': run.start_time.isoformat() if run.start_time else None,
            'end_time': run.end_time.isoformat() if run.end_time else None,
            'inputs': run.inputs,
            'outputs': run.outputs,
            'error': run.error,
            'parent_run_id': str(run.parent_run_id) if run.parent_run_id else None,
            'trace_id': str(run.trace_id) if run.trace_id else None,
            'dotted_order': run.dotted_order,
            'tags': run.tags,
            'extra': run.extra,
            'feedback_stats': run.feedback_stats,
            'status': getattr(run, 'status', None),
            'session_id': str(run.session_id) if hasattr(run, 'session_id') and run.session_id else None,
            'reference_example_id': str(run.reference_example_id) if hasattr(run, 'reference_example_id') and run.reference_example_id else None,
            # Token and cost information (if available)
            'total_tokens': getattr(run, 'total_tokens', None),
            'prompt_tokens': getattr(run, 'prompt_tokens', None),
            'completion_tokens': getattr(run, 'completion_tokens', None),
            'total_cost': str(getattr(run, 'total_cost', None)) if getattr(run, 'total_cost', None) else None,
            'prompt_cost': str(getattr(run, 'prompt_cost', None)) if getattr(run, 'prompt_cost', None) else None,
            'completion_cost': str(getattr(run, 'completion_cost', None)) if getattr(run, 'completion_cost', None) else None,
            'first_token_time': getattr(run, 'first_token_time', None).isoformat() if getattr(run, 'first_token_time', None) else None,
            # Child run information
            'child_run_ids': [str(cid) for cid in getattr(run, 'child_run_ids', [])] if getattr(run, 'child_run_ids', None) else [],
            # App path and other metadata
            'app_path': getattr(run, 'app_path', None),
            'in_dataset': getattr(run, 'in_dataset', None),
        }
        runs_data.append(run_dict)
    
    return runs_data

def save_traces_to_file(runs_data: List[Dict[str, Any]], 
                       filename: str, 
                       format_type: str = "json") -> None:
    """Save traces to file in specified format."""
    if format_type.lower() == "json":
        with open(f"{filename}.json", 'w') as f:
            json.dump(runs_data, f, indent=2, default=str)
        print(f"Traces saved to {filename}.json")
    
    elif format_type.lower() == "csv":
        if runs_data:
            with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                # Get all possible keys from all runs
                all_keys = set()
                for run in runs_data:
                    all_keys.update(run.keys())
                
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                
                for run in runs_data:
                    # Convert complex objects to strings for CSV
                    csv_row = {}
                    for key, value in run.items():
                        if isinstance(value, (dict, list)):
                            csv_row[key] = json.dumps(value) if value else ""
                        else:
                            csv_row[key] = str(value) if value is not None else ""
                    writer.writerow(csv_row)
            print(f"Traces saved to {filename}.csv")
    else:
        raise ValueError("format_type must be 'json' or 'csv'")

def print_run_summary(runs_data: List[Dict[str, Any]]) -> None:
    """Print a summary of downloaded runs."""
    if not runs_data:
        print("No runs to summarize")
        return
    
    print(f"\n📊 Run Summary:")
    print(f"  Total runs: {len(runs_data)}")
    
    # Run type breakdown
    run_types = {}
    for run in runs_data:
        rt = run.get('run_type', 'unknown')
        run_types[rt] = run_types.get(rt, 0) + 1
    
    print(f"  Run types:")
    for rt, count in sorted(run_types.items()):
        print(f"    {rt}: {count}")
    
    # Error summary
    error_count = sum(1 for run in runs_data if run.get('error'))
    print(f"  Runs with errors: {error_count}")
    
    # Token summary (for LLM runs)
    llm_runs = [run for run in runs_data if run.get('run_type') == 'llm']
    if llm_runs:
        total_tokens = sum(run.get('total_tokens', 0) or 0 for run in llm_runs)
        total_cost = sum(float(run.get('total_cost', 0) or 0) for run in llm_runs)
        print(f"  LLM stats:")
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Total cost: ${total_cost:.4f}")

def main():
    """Main function to demonstrate trace downloading."""
    try:
        # Load configuration from environment variables
        config = load_langsmith_config()
        
        print("🚀 LangSmith Trace Downloader")
        print("=" * 40)
        print(f"  Tracing: {config['tracing']}")
        print(f"  Endpoint: {config['endpoint']}")
        print(f"  Project: {config['project']}")
        print(f"  API Key: {'✓ Set' if config['api_key'] else '✗ Not set'}")
        print()
        
        # Example 1: Download all traces from the last 24 hours
        print("📥 Downloading traces from the last 24 hours...")
        start_time = datetime.now() - timedelta(days=1)
        runs_data = download_langsmith_traces(
            project_name=config['project'],
            api_key=config['api_key'],
            endpoint=config['endpoint'],
            start_time=start_time,
            limit=100  # Limit for demonstration
        )
        
        if runs_data:
            # Save as JSON
            save_traces_to_file(runs_data, "langsmith_traces_24h", "json")
            
            # Save as CSV
            save_traces_to_file(runs_data, "langsmith_traces_24h", "csv")
            
            # Print summary
            print_run_summary(runs_data)
            
            print(f"\n📝 Sample run data:")
            sample_run = runs_data[0]
            print(f"  Run ID: {sample_run['id']}")
            print(f"  Name: {sample_run['name']}")
            print(f"  Type: {sample_run['run_type']}")
            print(f"  Start Time: {sample_run['start_time']}")
            print(f"  Status: {sample_run.get('status', 'N/A')}")
        else:
            print("No traces found in the specified time range")
            
        # Example 2: Download only LLM runs
        print(f"\n📥 Downloading LLM runs...")
        llm_runs = download_langsmith_traces(
            project_name=config['project'],
            api_key=config['api_key'],
            endpoint=config['endpoint'],
            run_type="llm",
            limit=50
        )
        
        if llm_runs:
            save_traces_to_file(llm_runs, "langsmith_llm_traces", "json")
            print(f"✅ Downloaded {len(llm_runs)} LLM traces")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your .env file contains:")
        print("  LANGSMITH_TRACING=true")
        print("  LANGSMITH_ENDPOINT=https://api.smith.langchain.com")
        print("  LANGSMITH_API_KEY=your_api_key_here")
        print("  LANGSMITH_PROJECT=your_project_name")

if __name__ == "__main__":
    main()
