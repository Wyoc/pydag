"""Custom exceptions for PyDAG."""


class PyDAGError(Exception):
    """Base exception for all PyDAG errors.
    
    All PyDAG-specific exceptions inherit from this base class.
    """
    pass


class DAGValidationError(PyDAGError):
    """Raised when DAG validation fails.
    
    This exception is raised when the DAG structure is invalid,
    such as when required validation checks fail.
    """
    pass


class NodeNotFoundError(PyDAGError):
    """Raised when a referenced node does not exist.
    
    This typically occurs when trying to add dependencies to
    nodes that haven't been created yet.
    """
    pass


class CyclicGraphError(PyDAGError):
    """Raised when attempting to execute a cyclic graph.
    
    DAGs must be acyclic to be executable. This exception is
    raised when cycles are detected in the dependency graph.
    """
    pass


class DuplicateNodeError(PyDAGError):
    """Raised when attempting to add a node that already exists.
    
    Node names must be unique within a DAG. This exception is
    raised when trying to add a node with a name that's already taken.
    """
    pass


class ConditionEvaluationError(PyDAGError):
    """Raised when condition evaluation fails.
    
    This exception is raised when a node's condition function
    raises an exception during evaluation.
    """
    pass