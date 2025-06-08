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
    
    # Demonstrate parallel branches
    print("\n" + "="*50)
    print("PARALLEL BRANCH EXECUTION DEMO")
    print("="*50)
    
    run_parallel_branch_example()


def run_parallel_branch_example():
    """Demonstrate parallel branch execution capabilities."""
    # Create a DAG with distinct parallel branches
    branch_dag = DAG("parallel_branches_demo", max_workers=4)
    
    # Root task that starts everything
    @task(branch_dag, "data_source")
    def data_source():
        print("📊 Loading initial data...")
        time.sleep(0.5)
        return {
            "users": list(range(1000)),
            "orders": list(range(5000)),
            "products": list(range(500))
        }
    
    # Branch 1: User processing pipeline
    @task(branch_dag, "process_users", dependencies=["data_source"])
    def process_users():
        print("👥 Processing user data...")
        time.sleep(1.0)  # Simulate processing time
        data = branch_dag.nodes["data_source"].result
        processed = len(data["users"]) * 2
        print(f"👥 Processed {processed} user records")
        return processed
    
    @task(branch_dag, "user_analytics", dependencies=["process_users"])
    def user_analytics():
        print("📈 Running user analytics...")
        time.sleep(0.8)
        user_count = branch_dag.nodes["process_users"].result
        analytics = {"active_users": user_count // 2, "new_users": user_count // 4}
        print(f"📈 Analytics: {analytics}")
        return analytics
    
    # Branch 2: Order processing pipeline
    @task(branch_dag, "process_orders", dependencies=["data_source"])
    def process_orders():
        print("🛒 Processing order data...")
        time.sleep(1.2)  # Different processing time
        data = branch_dag.nodes["data_source"].result
        processed = len(data["orders"]) * 1.5
        print(f"🛒 Processed {processed} order records")
        return processed
    
    @task(branch_dag, "order_analytics", dependencies=["process_orders"])
    def order_analytics():
        print("💰 Running order analytics...")
        time.sleep(0.6)
        order_count = branch_dag.nodes["process_orders"].result
        analytics = {"total_revenue": order_count * 25, "avg_order": 25}
        print(f"💰 Analytics: {analytics}")
        return analytics
    
    # Branch 3: Product processing pipeline
    @task(branch_dag, "process_products", dependencies=["data_source"])
    def process_products():
        print("🏷️  Processing product data...")
        time.sleep(0.9)
        data = branch_dag.nodes["data_source"].result
        processed = len(data["products"]) * 3
        print(f"🏷️  Processed {processed} product records")
        return processed
    
    @task(branch_dag, "product_recommendations", dependencies=["process_products"])
    def product_recommendations():
        print("🎯 Generating product recommendations...")
        time.sleep(0.7)
        product_count = branch_dag.nodes["process_products"].result
        recommendations = {"trending": product_count // 10, "personalized": product_count // 5}
        print(f"🎯 Recommendations: {recommendations}")
        return recommendations
    
    # Convergence point - combines all analytics
    @task(branch_dag, "generate_dashboard", dependencies=["user_analytics", "order_analytics", "product_recommendations"])
    def generate_dashboard():
        print("📋 Generating final dashboard...")
        time.sleep(0.3)
        
        user_data = branch_dag.nodes["user_analytics"].result
        order_data = branch_dag.nodes["order_analytics"].result
        product_data = branch_dag.nodes["product_recommendations"].result
        
        dashboard = {
            "users": user_data,
            "orders": order_data,
            "products": product_data,
            "generated_at": time.time()
        }
        print("📋 Dashboard generated successfully!")
        return dashboard
    
    # Show parallel branches detected
    branches = branch_dag.identify_parallel_branches()
    print(f"\n🔍 Detected {len(branches)} parallel branches:")
    for i, branch in enumerate(branches, 1):
        print(f"   Branch {i}: {' → '.join(branch)}")
    
    print("\n⚡ Executing with parallel branch optimization...")
    start_time = time.time()
    
    # Execute with parallel branches (fallback to regular if no branches detected)
    if len(branches) > 1:
        results = branch_dag.execute(parallel_branches=True)
    else:
        print("Note: Falling back to regular async execution")
        results = branch_dag.execute(async_execution=True)
    
    parallel_time = time.time() - start_time
    print(f"\n✅ Parallel execution completed in {parallel_time:.2f} seconds")
    
    # Reset for comparison
    for node in branch_dag.nodes.values():
        node.status = node.status.__class__.PENDING
        node.result = None
        node.error = None
        node.execution_time = 0.0
    
    print("\n🐌 Executing with regular async mode for comparison...")
    start_time = time.time()
    
    # Execute with regular async
    results_regular = branch_dag.execute(async_execution=True, parallel_branches=False)
    
    regular_time = time.time() - start_time
    print(f"✅ Regular execution completed in {regular_time:.2f} seconds")
    
    # Show performance improvement
    improvement = ((regular_time - parallel_time) / regular_time) * 100
    print(f"\n🚀 Performance improvement: {improvement:.1f}% faster with parallel branches")
    
    # Show final dashboard
    print("\n📊 Final Dashboard:")
    dashboard = results.get("generate_dashboard", {})
    for section, data in dashboard.items():
        if isinstance(data, dict):
            print(f"   {section.title()}: {data}")


if __name__ == "__main__":
    run_examples()