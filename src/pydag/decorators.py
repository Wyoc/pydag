"""Decorators for PyDAG."""

from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dag import DAG
    from .core.retry import RetryConfig


def task(dag: "DAG", name: str, dependencies: Optional[List[str]] = None, condition: Optional[Callable[[], bool]] = None, retry_config: Optional["RetryConfig"] = None):
    """
    Decorator to add a function as a task to a DAG.
    
    Args:
        dag: The DAG to add the task to
        name: Name of the task
        dependencies: List of dependency task names
        condition: Optional condition function that must return True for the task to execute
        retry_config: Optional retry configuration for failed executions
        
    Returns:
        Decorator function
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