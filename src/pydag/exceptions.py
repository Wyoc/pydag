"""Custom exceptions for PyDAG."""


class PyDAGError(Exception):
    """Base exception for all PyDAG errors."""
    pass


class DAGValidationError(PyDAGError):
    """Raised when DAG validation fails."""
    pass


class NodeNotFoundError(PyDAGError):
    """Raised when a referenced node does not exist."""
    pass


class CyclicGraphError(PyDAGError):
    """Raised when attempting to execute a cyclic graph."""
    pass


class DuplicateNodeError(PyDAGError):
    """Raised when attempting to add a node that already exists."""
    pass


class ConditionEvaluationError(PyDAGError):
    """Raised when condition evaluation fails."""
    pass