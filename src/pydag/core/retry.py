"""Retry utilities for PyDAG."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional, Type, Tuple, Union


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 1
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    
    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")


@dataclass
class RetryStats:
    """Statistics about retry attempts."""
    attempt_count: int = 0
    total_delay: float = 0.0
    last_exception: Optional[Exception] = None
    exceptions: list = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = []


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for a retry attempt with exponential backoff.
    
    Args:
        attempt: Current attempt number (1-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if attempt <= 1:
        return 0.0
    
    # Calculate exponential delay
    delay = config.base_delay * (config.exponential_base ** (attempt - 2))
    
    # Cap at max_delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        # Add +/- 10% jitter
        jitter_range = delay * 0.1
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay


def should_retry(exception: Exception, config: RetryConfig) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    Args:
        exception: The exception that occurred
        config: Retry configuration
        
    Returns:
        True if the exception should trigger a retry
    """
    if config.retryable_exceptions is None:
        # Retry all exceptions by default
        return True
    
    return isinstance(exception, config.retryable_exceptions)


async def execute_with_retry_async(
    func, 
    config: RetryConfig, 
    logger: Optional[logging.Logger] = None,
    *args, 
    **kwargs
) -> Tuple[any, RetryStats]:
    """
    Execute an async function with retry logic.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        logger: Optional logger for retry messages
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Tuple of (result, retry_stats)
        
    Raises:
        Exception: The last exception if all retries are exhausted
    """
    stats = RetryStats()
    
    for attempt in range(1, config.max_attempts + 1):
        stats.attempt_count = attempt
        
        try:
            if logger and attempt > 1:
                logger.info(f"Retry attempt {attempt}/{config.max_attempts}")
            
            result = await func(*args, **kwargs)
            return result, stats
            
        except Exception as e:
            stats.last_exception = e
            stats.exceptions.append(e)
            
            # Check if we should retry this exception
            if not should_retry(e, config):
                if logger:
                    logger.error(f"Non-retryable exception: {e}")
                raise
            
            # Check if we have more attempts
            if attempt >= config.max_attempts:
                if logger:
                    logger.error(f"All {config.max_attempts} retry attempts exhausted")
                # Attach stats to the exception for retrieval
                e.__retry_stats__ = stats
                raise
            
            # Calculate and apply delay
            delay = calculate_delay(attempt + 1, config)
            stats.total_delay += delay
            
            if logger:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
            
            if delay > 0:
                await asyncio.sleep(delay)


def execute_with_retry_sync(
    func, 
    config: RetryConfig, 
    logger: Optional[logging.Logger] = None,
    *args, 
    **kwargs
) -> Tuple[any, RetryStats]:
    """
    Execute a sync function with retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration
        logger: Optional logger for retry messages
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Tuple of (result, retry_stats)
        
    Raises:
        Exception: The last exception if all retries are exhausted
    """
    stats = RetryStats()
    
    for attempt in range(1, config.max_attempts + 1):
        stats.attempt_count = attempt
        
        try:
            if logger and attempt > 1:
                logger.info(f"Retry attempt {attempt}/{config.max_attempts}")
            
            result = func(*args, **kwargs)
            return result, stats
            
        except Exception as e:
            stats.last_exception = e
            stats.exceptions.append(e)
            
            # Check if we should retry this exception
            if not should_retry(e, config):
                if logger:
                    logger.error(f"Non-retryable exception: {e}")
                raise
            
            # Check if we have more attempts
            if attempt >= config.max_attempts:
                if logger:
                    logger.error(f"All {config.max_attempts} retry attempts exhausted")
                # Attach stats to the exception for retrieval
                e.__retry_stats__ = stats
                raise
            
            # Calculate and apply delay
            delay = calculate_delay(attempt + 1, config)
            stats.total_delay += delay
            
            if logger:
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
            
            if delay > 0:
                time.sleep(delay)