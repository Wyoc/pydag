"""Node and NodeStatus definitions for PyDAG."""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Set

from .retry import RetryConfig, RetryStats, execute_with_retry_async, execute_with_retry_sync


class NodeStatus(Enum):
    """Status of a node in the DAG execution.
    
    Attributes:
        PENDING: Node is waiting to be executed.
        RUNNING: Node is currently executing.
        COMPLETED: Node has completed successfully.
        FAILED: Node execution failed with an exception.
        SKIPPED: Node was skipped due to failed dependencies.
        CONDITION_NOT_MET: Node was skipped because its condition evaluated to False.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONDITION_NOT_MET = "condition_not_met"


@dataclass
class Node:
    """A node in the DAG representing a function to be executed.
    
    Each node wraps a function along with its execution metadata, dependencies,
    and configuration. The node handles both synchronous and asynchronous functions,
    retry logic, conditional execution, and status tracking.
    
    Attributes:
        name: Unique identifier for the node.
        func: The function to execute (can be sync or async).
        dependencies: Set of node names this node depends on.
        condition: Optional function that must return True for execution.
        retry_config: Configuration for retry behavior on failures.
        status: Current execution status of the node.
        result: Result value from successful execution.
        error: Exception from failed execution.
        execution_time: Total time taken for execution in seconds.
        retry_stats: Statistics about retry attempts.
    """
    name: str
    func: Callable
    dependencies: Set[str] = field(default_factory=set)
    condition: Optional[Callable[[], bool]] = None
    retry_config: Optional[RetryConfig] = None
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_stats: Optional[RetryStats] = None
    
    def __post_init__(self):
        # Store original function 
        self.original_func = self.func
        self.is_async = inspect.iscoroutinefunction(self.original_func)
        
        # Set default retry config if none provided
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        
        # Initialize retry stats
        self.retry_stats = RetryStats()
        
        # Create logger for this node
        self.logger = logging.getLogger(f"dagexec.node.{self.name}")
        
        if self.is_async:
            # Create async wrapped function
            @wraps(self.original_func)
            async def async_wrapped_func(*args, **kwargs):
                start_time = time.time()
                try:
                    self.status = NodeStatus.RUNNING
                    
                    if self.retry_config.max_attempts > 1:
                        # Execute with retry
                        try:
                            result, retry_stats = await execute_with_retry_async(
                                self.original_func, 
                                self.retry_config, 
                                self.logger,
                                *args, 
                                **kwargs
                            )
                            self.retry_stats = retry_stats
                        except Exception as e:
                            # Preserve retry stats even on failure
                            if hasattr(e, '__retry_stats__'):
                                self.retry_stats = e.__retry_stats__
                            raise
                    else:
                        # Execute once without retry
                        result = await self.original_func(*args, **kwargs)
                        self.retry_stats.attempt_count = 1
                    
                    self.result = result
                    self.status = NodeStatus.COMPLETED
                    return result
                except Exception as e:
                    self.error = e
                    self.status = NodeStatus.FAILED
                    # Preserve retry stats even when not retrying
                    if self.retry_config.max_attempts == 1:
                        self.retry_stats.attempt_count = 1
                        self.retry_stats.last_exception = e
                        self.retry_stats.exceptions.append(e)
                    raise
                finally:
                    self.execution_time = time.time() - start_time
            
            self.func = async_wrapped_func
        else:
            # Create sync wrapped function
            @wraps(self.original_func)
            def wrapped_func(*args, **kwargs):
                start_time = time.time()
                try:
                    self.status = NodeStatus.RUNNING
                    
                    if self.retry_config.max_attempts > 1:
                        # Execute with retry
                        try:
                            result, retry_stats = execute_with_retry_sync(
                                self.original_func, 
                                self.retry_config, 
                                self.logger,
                                *args, 
                                **kwargs
                            )
                            self.retry_stats = retry_stats
                        except Exception as e:
                            # Preserve retry stats even on failure
                            if hasattr(e, '__retry_stats__'):
                                self.retry_stats = e.__retry_stats__
                            raise
                    else:
                        # Execute once without retry
                        result = self.original_func(*args, **kwargs)
                        self.retry_stats.attempt_count = 1
                    
                    self.result = result
                    self.status = NodeStatus.COMPLETED
                    return result
                except Exception as e:
                    self.error = e
                    self.status = NodeStatus.FAILED
                    # Preserve retry stats even when not retrying
                    if self.retry_config.max_attempts == 1:
                        self.retry_stats.attempt_count = 1
                        self.retry_stats.last_exception = e
                        self.retry_stats.exceptions.append(e)
                    raise
                finally:
                    self.execution_time = time.time() - start_time
            
            self.func = wrapped_func