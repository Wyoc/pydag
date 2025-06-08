"""Decorators for PyDAG."""

from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dag import DAG
    from .core.retry import RetryConfig


def task(dag: "DAG", name: str, dependencies: Optional[List[str]] = None, condition: Optional[Callable[[], bool]] = None, retry_config: Optional["RetryConfig"] = None):
    """Decorator to add a function as a task to a DAG.
    
    This decorator provides a convenient way to register functions as DAG nodes
    without explicitly calling dag.add_node(). The decorated function can be
    either synchronous or asynchronous.
    
    Args:
        dag: The DAG instance to add the task to.
        name: Unique name for the task within the DAG.
        dependencies: List of task names this task depends on.
            The task will only execute after all dependencies complete successfully.
        condition: Optional function that must return True for the task to execute.
            If False, the task will be skipped.
        retry_config: Optional retry configuration for handling failed executions.
        
    Returns:
        Decorator function that registers the wrapped function as a DAG task.
        
    Example:
        >>> dag = DAG("example")
        >>> @task(dag, "process_data", dependencies=["load_data"])
        ... def process_data():
        ...     return "processed"
    """
    def decorator(func):
        # Add the function as a node to the DAG
        node = dag.add_node(name, func, dependencies, condition, retry_config)
        
        # Create a property on the function to access the node's result
        # This allows direct access like function_name.result
        class FunctionWithResult:
            @property
            def result(self_inner):
                return dag.nodes[name].result
                
        # Add result property to the function
        func.result = property(lambda: dag.nodes[name].result)
        
        return func
    return decorator