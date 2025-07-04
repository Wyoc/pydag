# PyDAG

A Python package for executing functions as a Directed Acyclic Graph (DAG) with built-in concurrency support, retry logic, and conditional execution.

## Features

- **DAG Execution**: Execute functions in dependency order with automatic parallelization
- **Sync/Async Support**: Mix synchronous and asynchronous tasks seamlessly  
- **Retry Logic**: Automatic retry with exponential backoff for failed tasks
- **Conditional Execution**: Skip tasks based on runtime conditions
- **Visualization**: Built-in graph visualization with NetworkX and Matplotlib
- **Concurrency Control**: Configurable worker pools for parallel execution
- **Status Tracking**: Real-time task status monitoring (PENDING � RUNNING � COMPLETED/FAILED/SKIPPED)

## Installation

```bash
# Install the package
uv sync

# Install with test dependencies
uv sync --extra test
```

## Quick Start

```python
from pydag import DAG, task

# Create a DAG
dag = DAG("my_workflow", max_workers=4)

# Define tasks with dependencies
@task(dag, "fetch_data")
def fetch_data():
    return {"users": [1, 2, 3], "products": [10, 20, 30]}

@task(dag, "process_users", dependencies=["fetch_data"])
def process_users():
    data = dag.nodes["fetch_data"].result
    return [u * 2 for u in data["users"]]

@task(dag, "process_products", dependencies=["fetch_data"])  
def process_products():
    data = dag.nodes["fetch_data"].result
    return [p * 1.1 for p in data["products"]]

@task(dag, "combine_results", dependencies=["process_users", "process_products"])
def combine_results():
    users = dag.nodes["process_users"].result
    products = dag.nodes["process_products"].result
    return {"processed_users": users, "processed_products": products}

# Execute the DAG
results = dag.execute()
print(results["combine_results"])
```

## Advanced Features

### Retry Logic

```python
from pydag import RetryConfig

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    exponential_base=2.0,
    max_delay=10.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
)

@task(dag, "flaky_network_call", retry_config=retry_config)
def flaky_network_call():
    # This will retry up to 3 times with exponential backoff
    return make_api_request()
```

### Conditional Execution

```python
def should_run_expensive_task():
    return os.getenv("RUN_EXPENSIVE") == "true"

@task(dag, "expensive_task", condition=should_run_expensive_task)
def expensive_task():
    # Only runs if condition returns True
    return perform_expensive_computation()
```

### Async Tasks

```python
@task(dag, "async_fetch")
async def async_fetch():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            return await response.json()

@task(dag, "process_async_data", dependencies=["async_fetch"])
async def process_async_data():
    data = dag.nodes["async_fetch"].result
    return await process_data_async(data)
```

### Visualization

```python
# Visualize DAG structure
dag.visualize()

# Execute and visualize with status colors
results = dag.execute()
dag.visualize()  # Shows completed/failed/skipped status
```

## API Reference

### DAG Class

```python
DAG(name: str, max_workers: int = 4)
```

- `add_node(name, func, dependencies=None, condition=None, retry_config=None)`
- `execute(**kwargs)` - Execute all tasks and return results
- `visualize()` - Generate visual representation of the DAG

### Task Decorator

```python
@task(dag, name, dependencies=None, condition=None, retry_config=None)
```

Convenient decorator to add functions as DAG nodes.

### RetryConfig

```python
RetryConfig(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,)
)
```

## Development

### Run Examples

```bash
uv run python -m pydag.examples
```

### Run Tests

```bash
uv run pytest tests/ -v
```

### Project Structure

```
src/pydag/
   __init__.py              # Main exports
   dag.py                   # Core DAG execution engine
   decorators.py            # @task decorator
   exceptions.py            # Custom exceptions
   visualization.py         # Graph visualization
   examples.py             # Usage examples
   core/
       node.py             # Node class and status tracking
       retry.py            # Retry logic implementation
```

## Requirements

- Python e 3.13
- NetworkX (graph operations)
- Matplotlib (visualization)

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.