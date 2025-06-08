"""Tests for retry functionality in PyDAG."""

import pytest
import asyncio
import time
from unittest.mock import patch

from pydag import DAG, RetryConfig, RetryStats, task
from pydag.core.retry import calculate_delay, should_retry, execute_with_retry_sync, execute_with_retry_async


class TestRetryConfig:
    def test_default_config(self):
        config = RetryConfig()
        assert config.max_attempts == 1
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions is None

    def test_custom_config(self):
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, RuntimeError)
        )
        assert config.max_attempts == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, RuntimeError)

    def test_invalid_config_validation(self):
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryConfig(max_attempts=0)
        
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            RetryConfig(base_delay=-1)
        
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryConfig(base_delay=10, max_delay=5)
        
        with pytest.raises(ValueError, match="exponential_base must be > 1"):
            RetryConfig(exponential_base=1.0)


class TestRetryCalculations:
    def test_calculate_delay_first_attempt(self):
        config = RetryConfig()
        delay = calculate_delay(1, config)
        assert delay == 0.0

    def test_calculate_delay_exponential(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert calculate_delay(2, config) == 1.0  # 1 * 2^0
        assert calculate_delay(3, config) == 2.0  # 1 * 2^1  
        assert calculate_delay(4, config) == 4.0  # 1 * 2^2

    def test_calculate_delay_max_cap(self):
        config = RetryConfig(base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        
        # Should cap at max_delay
        delay = calculate_delay(10, config)
        assert delay == 5.0

    def test_calculate_delay_with_jitter(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=True)
        
        # Test jitter adds randomness
        delays = [calculate_delay(2, config) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have different values due to jitter
        
        # All delays should be around base_delay +/- 10%
        for delay in delays:
            assert 0.9 <= delay <= 1.1

    def test_should_retry_all_exceptions(self):
        config = RetryConfig()
        
        assert should_retry(ValueError("test"), config) is True
        assert should_retry(RuntimeError("test"), config) is True
        assert should_retry(Exception("test"), config) is True

    def test_should_retry_specific_exceptions(self):
        config = RetryConfig(retryable_exceptions=(ValueError, RuntimeError))
        
        assert should_retry(ValueError("test"), config) is True
        assert should_retry(RuntimeError("test"), config) is True
        assert should_retry(TypeError("test"), config) is False


class TestRetryExecution:
    def test_sync_retry_success_first_attempt(self):
        config = RetryConfig(max_attempts=3)
        
        def success_func():
            return "success"
        
        result, stats = execute_with_retry_sync(success_func, config)
        
        assert result == "success"
        assert stats.attempt_count == 1
        assert stats.total_delay == 0.0
        assert stats.last_exception is None
        assert len(stats.exceptions) == 0

    def test_sync_retry_success_after_failures(self):
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"
        
        result, stats = execute_with_retry_sync(flaky_func, config)
        
        assert result == "success"
        assert stats.attempt_count == 3
        assert stats.total_delay > 0
        assert len(stats.exceptions) == 2

    def test_sync_retry_all_attempts_fail(self):
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        def failing_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            execute_with_retry_sync(failing_func, config)

    @pytest.mark.asyncio
    async def test_async_retry_success_first_attempt(self):
        config = RetryConfig(max_attempts=3)
        
        async def success_func():
            return "success"
        
        result, stats = await execute_with_retry_async(success_func, config)
        
        assert result == "success"
        assert stats.attempt_count == 1
        assert stats.total_delay == 0.0

    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"
        
        result, stats = await execute_with_retry_async(flaky_func, config)
        
        assert result == "success"
        assert stats.attempt_count == 3
        assert stats.total_delay > 0

    @pytest.mark.asyncio
    async def test_async_retry_all_attempts_fail(self):
        config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        async def failing_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            await execute_with_retry_async(failing_func, config)

    def test_sync_retry_non_retryable_exception(self):
        config = RetryConfig(max_attempts=3, retryable_exceptions=(ValueError,))
        
        def failing_func():
            raise TypeError("Non-retryable")
        
        with pytest.raises(TypeError, match="Non-retryable"):
            execute_with_retry_sync(failing_func, config)


class TestDAGRetryIntegration:
    def test_node_with_retry_success(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        def test_func():
            return 42
        
        dag.add_node("retry_node", test_func, retry_config=retry_config)
        
        results = dag.execute_sync()
        
        assert results["retry_node"] == 42
        node = dag.nodes["retry_node"]
        assert node.retry_stats.attempt_count == 1

    def test_node_with_retry_after_failures(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        
        call_count = 0
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return 42
        
        dag.add_node("retry_node", flaky_func, retry_config=retry_config)
        
        results = dag.execute_sync()
        
        assert results["retry_node"] == 42
        node = dag.nodes["retry_node"]
        assert node.retry_stats.attempt_count == 3
        assert len(node.retry_stats.exceptions) == 2

    def test_node_retry_all_attempts_fail(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        def failing_func():
            raise ValueError("Always fails")
        
        dag.add_node("retry_node", failing_func, retry_config=retry_config)
        
        results = dag.execute_sync()
        
        assert "retry_node" not in results
        node = dag.nodes["retry_node"]
        assert node.retry_stats.attempt_count == 2
        assert isinstance(node.error, ValueError)

    def test_task_decorator_with_retry(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        
        call_count = 0
        
        @task(dag, "retry_task", retry_config=retry_config)
        def retry_task():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError(f"Attempt {call_count}")
            return 100
        
        results = dag.execute_sync()
        
        assert results["retry_task"] == 100
        node = dag.nodes["retry_task"]
        assert node.retry_stats.attempt_count == 2

    @pytest.mark.asyncio
    async def test_async_node_with_retry(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        
        call_count = 0
        async def flaky_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError(f"Async attempt {call_count}")
            return 200
        
        dag.add_node("async_retry_node", flaky_async_func, retry_config=retry_config)
        
        results = await dag.execute_async()
        
        assert results["async_retry_node"] == 200
        node = dag.nodes["async_retry_node"]
        assert node.retry_stats.attempt_count == 2

    def test_retry_with_dependencies(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        
        def success_func():
            return 10
        
        call_count = 0
        def flaky_dependent():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Dependency retry")
            return dag.nodes["success_node"].result * 2
        
        dag.add_node("success_node", success_func)
        dag.add_node("retry_dependent", flaky_dependent, 
                    dependencies=["success_node"], retry_config=retry_config)
        
        results = dag.execute_sync()
        
        assert results["success_node"] == 10
        assert results["retry_dependent"] == 20
        assert dag.nodes["retry_dependent"].retry_stats.attempt_count == 2

    def test_retry_timing_accuracy(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=2, base_delay=0.2, jitter=False)
        
        def failing_func():
            raise ValueError("Timing test")
        
        dag.add_node("timing_node", failing_func, retry_config=retry_config)
        
        start_time = time.time()
        results = dag.execute_sync()
        end_time = time.time()
        
        execution_time = end_time - start_time
        node = dag.nodes["timing_node"]
        
        # Should have taken at least the retry delay
        assert execution_time >= 0.2
        assert node.retry_stats.total_delay >= 0.2


class TestRetryEdgeCases:
    def test_no_retry_config_defaults_to_single_attempt(self):
        dag = DAG("test_dag")
        
        def test_func():
            return 42
        
        dag.add_node("no_retry_node", test_func)  # No retry_config
        
        results = dag.execute_sync()
        
        assert results["no_retry_node"] == 42
        node = dag.nodes["no_retry_node"]
        assert node.retry_stats.attempt_count == 1

    def test_retry_config_max_attempts_one(self):
        dag = DAG("test_dag")
        retry_config = RetryConfig(max_attempts=1)  # Explicit single attempt
        
        def failing_func():
            raise ValueError("Should not retry")
        
        dag.add_node("single_attempt", failing_func, retry_config=retry_config)
        
        results = dag.execute_sync()
        
        assert "single_attempt" not in results
        node = dag.nodes["single_attempt"]
        assert node.retry_stats.attempt_count == 1
        assert isinstance(node.error, ValueError)