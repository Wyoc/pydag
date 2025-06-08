"""Example usage of PyDAG."""

import asyncio
import logging
import time

from . import DAG, task, RetryConfig


def run_examples():
    """Run example DAG executions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a DAG
    dag = DAG("example", max_workers=4)
    
    # Define sync tasks
    @task(dag, "task1")
    def task1(x=1):
        print(f"Executing task1 with x={x}")
        time.sleep(2)
        return x + 1
    
    @task(dag, "task2")
    def task2(x=1):
        print(f"Executing task2 with x={x}")
        time.sleep(3)
        return x + 2
    
    @task(dag, "task3", dependencies=["task1"])
    def task3(x=1):
        print(f"Executing task3 with x={x}")
        time.sleep(1)
        task1_node = dag.nodes["task1"]
        return task1_node.result * 2
    
    @task(dag, "task4", dependencies=["task2"])
    def task4(x=1):
        print(f"Executing task4 with x={x}")
        time.sleep(2)
        task2_node = dag.nodes["task2"]
        return task2_node.result * 2
    
    @task(dag, "task5", dependencies=["task3", "task4"])
    def task5(x=1):
        print(f"Executing task5 with x={x}")
        time.sleep(1)
        task3_node = dag.nodes["task3"]
        task4_node = dag.nodes["task4"]
        return task3_node.result + task4_node.result
    
    # Example async tasks
    @task(dag, "async_task1")
    async def async_task1(x=1):
        print(f"Executing async_task1 with x={x}")
        await asyncio.sleep(2)  # Simulate async I/O
        return x * 10
    
    @task(dag, "async_task2", dependencies=["async_task1"])
    async def async_task2(x=1):
        print(f"Executing async_task2 with x={x}")
        await asyncio.sleep(1)
        async_task1_node = dag.nodes["async_task1"]
        return async_task1_node.result + 5
    
    # Example conditional tasks
    def condition_check():
        return True  # This could be any logic
    
    def expensive_condition():
        return False  # Simulate expensive condition that evaluates to False
    
    @task(dag, "conditional_task1", condition=condition_check)
    def conditional_task1(x=1):
        print(f"Executing conditional_task1 with x={x}")
        time.sleep(1)
        return x + 100
    
    @task(dag, "expensive_task", condition=expensive_condition)
    def expensive_task(x=1):
        print(f"This expensive task should not run")
        time.sleep(5)  # This won't execute due to condition
        return x * 1000
    
    # Example retry tasks
    retry_config = RetryConfig(max_attempts=3, base_delay=0.5, exponential_base=2.0)
    
    # Use a closure to maintain state
    def make_flaky_task():
        call_count = 0
        def flaky_task(x=1):
            nonlocal call_count
            call_count += 1
            print(f"Flaky task attempt {call_count}")
            if call_count < 3:  # Fail first 2 attempts
                raise RuntimeError(f"Simulated failure on attempt {call_count}")
            return x + 500
        return flaky_task
    
    dag.add_node("flaky_task", make_flaky_task(), retry_config=retry_config)
    
    # Network-like task that might fail
    network_retry_config = RetryConfig(
        max_attempts=4, 
        base_delay=0.2, 
        max_delay=5.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    
    def make_network_task():
        call_count = 0
        def network_task(x=1):
            nonlocal call_count
            call_count += 1
            print(f"Network task attempt {call_count}")
            if call_count < 2:  # Fail first attempt
                raise ConnectionError("Network unavailable")
            return x + 1000
        return network_task
    
    dag.add_node("network_task", make_network_task(), retry_config=network_retry_config)
    
    # Visualize the DAG
    dag.visualize()
    
    # Execute the DAG
    results = dag.execute(x=10)
    print("Results:", results)
    
    # Print retry statistics
    print("\nRetry Statistics:")
    for name, node in dag.nodes.items():
        if node.retry_stats and node.retry_stats.attempt_count > 1:
            print(f"  {name}: {node.retry_stats.attempt_count} attempts, "
                  f"{node.retry_stats.total_delay:.2f}s total delay")
    
    # Visualize the executed DAG
    dag.visualize()


if __name__ == "__main__":
    run_examples()