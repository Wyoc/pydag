"""
PyDAG - A package for executing functions as a Directed Acyclic Graph (DAG) with concurrency.
"""

from .core.node import Node, NodeStatus
from .core.retry import RetryConfig, RetryStats
from .dag import DAG
from .decorators import task
from .exceptions import (
    PyDAGError,
    DAGValidationError,
    NodeNotFoundError,
    CyclicGraphError,
    DuplicateNodeError,
    ConditionEvaluationError
)

__version__ = "0.1.0"

__all__ = [
    "DAG",
    "Node", 
    "NodeStatus",
    "RetryConfig",
    "RetryStats",
    "task",
    "PyDAGError",
    "DAGValidationError", 
    "NodeNotFoundError",
    "CyclicGraphError",
    "DuplicateNodeError",
    "ConditionEvaluationError"
]