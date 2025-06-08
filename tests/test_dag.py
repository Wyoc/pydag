import pytest
import asyncio
import time
from pydag import DAG, Node, NodeStatus, task
from pydag.exceptions import DuplicateNodeError, NodeNotFoundError, CyclicGraphError


class TestBasicDAG:
    def test_dag_creation(self):
        dag = DAG("test_dag")
        assert dag.name == "test_dag"
        assert len(dag.nodes) == 0
        assert dag.max_workers is None

    def test_add_sync_node(self):
        dag = DAG("test_dag")
        
        def test_func():
            return 42
        
        node = dag.add_node("test_node", test_func)
        assert node.name == "test_node"
        assert node.status == NodeStatus.PENDING
        assert not node.is_async
        assert len(dag.nodes) == 1

    def test_add_node_with_dependencies(self):
        dag = DAG("test_dag")
        
        def func1():
            return 1
        
        def func2():
            return 2
        
        dag.add_node("node1", func1)
        dag.add_node("node2", func2, dependencies=["node1"])
        
        assert "node1" in dag.nodes["node2"].dependencies
        assert len(dag.nodes["node2"].dependencies) == 1

    def test_duplicate_node_raises_error(self):
        dag = DAG("test_dag")
        
        def test_func():
            return 42
        
        dag.add_node("test_node", test_func)
        
        with pytest.raises(DuplicateNodeError, match="Node 'test_node' already exists"):
            dag.add_node("test_node", test_func)

    def test_nonexistent_dependency_raises_error(self):
        dag = DAG("test_dag")
        
        def test_func():
            return 42
        
        with pytest.raises(NodeNotFoundError, match="Dependency 'nonexistent' does not exist"):
            dag.add_node("test_node", test_func, dependencies=["nonexistent"])

    def test_simple_sync_execution(self):
        dag = DAG("test_dag")
        
        def func1():
            return 10
        
        def func2():
            return 20
        
        dag.add_node("node1", func1)
        dag.add_node("node2", func2)
        
        results = dag.execute_sync()
        
        assert results["node1"] == 10
        assert results["node2"] == 20
        assert dag.nodes["node1"].status == NodeStatus.COMPLETED
        assert dag.nodes["node2"].status == NodeStatus.COMPLETED

    def test_sync_execution_with_dependencies(self):
        dag = DAG("test_dag")
        
        def func1():
            return 10
        
        def func2():
            node1 = dag.nodes["node1"]
            return node1.result * 2
        
        dag.add_node("node1", func1)
        dag.add_node("node2", func2, dependencies=["node1"])
        
        results = dag.execute_sync()
        
        assert results["node1"] == 10
        assert results["node2"] == 20

    def test_task_decorator(self):
        dag = DAG("test_dag")
        
        @task(dag, "task1")
        def task1():
            return 42
        
        @task(dag, "task2", dependencies=["task1"])
        def task2():
            return dag.nodes["task1"].result * 2
        
        results = dag.execute_sync()
        
        assert results["task1"] == 42
        assert results["task2"] == 84


class TestAsyncDAG:
    def test_add_async_node(self):
        dag = DAG("test_dag")
        
        async def async_func():
            return 42
        
        node = dag.add_node("async_node", async_func)
        assert node.name == "async_node"
        assert node.is_async
        assert node.status == NodeStatus.PENDING

    def test_simple_async_execution_sync_mode(self):
        dag = DAG("test_dag")
        
        async def async_func():
            await asyncio.sleep(0.1)
            return 42
        
        dag.add_node("async_node", async_func)
        
        results = dag.execute_sync()
        
        assert results["async_node"] == 42
        assert dag.nodes["async_node"].status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_simple_async_execution_async_mode(self):
        dag = DAG("test_dag")
        
        async def async_func():
            await asyncio.sleep(0.1)
            return 42
        
        dag.add_node("async_node", async_func)
        
        results = await dag.execute_async()
        
        assert results["async_node"] == 42
        assert dag.nodes["async_node"].status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_dependencies(self):
        dag = DAG("test_dag")
        
        async def async_func1():
            await asyncio.sleep(0.1)
            return 10
        
        async def async_func2():
            await asyncio.sleep(0.1)
            node1 = dag.nodes["async_node1"]
            return node1.result * 2
        
        dag.add_node("async_node1", async_func1)
        dag.add_node("async_node2", async_func2, dependencies=["async_node1"])
        
        results = await dag.execute_async()
        
        assert results["async_node1"] == 10
        assert results["async_node2"] == 20

    def test_async_task_decorator(self):
        dag = DAG("test_dag")
        
        @task(dag, "async_task1")
        async def async_task1():
            await asyncio.sleep(0.1)
            return 42
        
        @task(dag, "async_task2", dependencies=["async_task1"])
        async def async_task2():
            await asyncio.sleep(0.1)
            return dag.nodes["async_task1"].result * 2
        
        results = dag.execute_sync()
        
        assert results["async_task1"] == 42
        assert results["async_task2"] == 84


class TestMixedSyncAsync:
    def test_mixed_execution_sync_mode(self):
        dag = DAG("test_dag")
        
        def sync_func():
            return 10
        
        async def async_func():
            await asyncio.sleep(0.1)
            return 20
        
        def dependent_func():
            sync_result = dag.nodes["sync_node"].result
            async_result = dag.nodes["async_node"].result
            return sync_result + async_result
        
        dag.add_node("sync_node", sync_func)
        dag.add_node("async_node", async_func)
        dag.add_node("dependent_node", dependent_func, dependencies=["sync_node", "async_node"])
        
        results = dag.execute_sync()
        
        assert results["sync_node"] == 10
        assert results["async_node"] == 20
        assert results["dependent_node"] == 30

    @pytest.mark.asyncio
    async def test_mixed_execution_async_mode(self):
        dag = DAG("test_dag")
        
        def sync_func():
            time.sleep(0.1)
            return 10
        
        async def async_func():
            await asyncio.sleep(0.1)
            return 20
        
        def dependent_func():
            sync_result = dag.nodes["sync_node"].result
            async_result = dag.nodes["async_node"].result
            return sync_result + async_result
        
        dag.add_node("sync_node", sync_func)
        dag.add_node("async_node", async_func)
        dag.add_node("dependent_node", dependent_func, dependencies=["sync_node", "async_node"])
        
        results = await dag.execute_async()
        
        assert results["sync_node"] == 10
        assert results["async_node"] == 20
        assert results["dependent_node"] == 30

    def test_async_depends_on_sync(self):
        dag = DAG("test_dag")
        
        def sync_func():
            return 10
        
        async def async_func():
            await asyncio.sleep(0.1)
            sync_result = dag.nodes["sync_node"].result
            return sync_result * 2
        
        dag.add_node("sync_node", sync_func)
        dag.add_node("async_node", async_func, dependencies=["sync_node"])
        
        results = dag.execute_sync()
        
        assert results["sync_node"] == 10
        assert results["async_node"] == 20

    def test_sync_depends_on_async(self):
        dag = DAG("test_dag")
        
        async def async_func():
            await asyncio.sleep(0.1)
            return 10
        
        def sync_func():
            async_result = dag.nodes["async_node"].result
            return async_result * 2
        
        dag.add_node("async_node", async_func)
        dag.add_node("sync_node", sync_func, dependencies=["async_node"])
        
        results = dag.execute_sync()
        
        assert results["async_node"] == 10
        assert results["sync_node"] == 20


class TestErrorHandling:
    def test_sync_function_error(self):
        dag = DAG("test_dag")
        
        def failing_func():
            raise ValueError("Test error")
        
        def dependent_func():
            return 42
        
        dag.add_node("failing_node", failing_func)
        dag.add_node("dependent_node", dependent_func, dependencies=["failing_node"])
        
        results = dag.execute_sync()
        
        assert dag.nodes["failing_node"].status == NodeStatus.FAILED
        assert dag.nodes["dependent_node"].status == NodeStatus.SKIPPED
        assert "failing_node" not in results
        assert "dependent_node" not in results

    def test_async_function_error(self):
        dag = DAG("test_dag")
        
        async def failing_async_func():
            await asyncio.sleep(0.1)
            raise ValueError("Async test error")
        
        def dependent_func():
            return 42
        
        dag.add_node("failing_async_node", failing_async_func)
        dag.add_node("dependent_node", dependent_func, dependencies=["failing_async_node"])
        
        results = dag.execute_sync()
        
        assert dag.nodes["failing_async_node"].status == NodeStatus.FAILED
        assert dag.nodes["dependent_node"].status == NodeStatus.SKIPPED
        assert "failing_async_node" not in results
        assert "dependent_node" not in results

    @pytest.mark.asyncio
    async def test_async_mode_error_handling(self):
        dag = DAG("test_dag")
        
        async def failing_func():
            await asyncio.sleep(0.1)
            raise ValueError("Async error")
        
        async def dependent_func():
            await asyncio.sleep(0.1)
            return 42
        
        dag.add_node("failing_node", failing_func)
        dag.add_node("dependent_node", dependent_func, dependencies=["failing_node"])
        
        results = await dag.execute_async()
        
        assert dag.nodes["failing_node"].status == NodeStatus.FAILED
        assert dag.nodes["dependent_node"].status == NodeStatus.SKIPPED
        assert "failing_node" not in results
        assert "dependent_node" not in results


class TestCycleDetection:
    def test_verify_acyclic_valid_dag(self):
        dag = DAG("test_dag")
        
        def func1():
            return 1
        
        def func2():
            return 2
        
        def func3():
            return 3
        
        dag.add_node("node1", func1)
        dag.add_node("node2", func2, dependencies=["node1"])
        dag.add_node("node3", func3, dependencies=["node2"])
        
        assert dag.verify_acyclic() is True

    def test_execute_with_cycle_raises_error(self):
        dag = DAG("test_dag")
        
        def func1():
            return 1
        
        def func2():
            return 2
        
        dag.add_node("node1", func1)
        dag.add_node("node2", func2, dependencies=["node1"])
        
        # Manually create a cycle by adding edge to the graph
        dag._graph.add_edge("node2", "node1")
        
        with pytest.raises(CyclicGraphError, match="Cannot execute cyclic graph"):
            dag.execute_sync()


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(self):
        dag = DAG("test_dag", max_workers=4)
        
        async def slow_async_func(delay=0.2):
            await asyncio.sleep(delay)
            return delay
        
        def slow_sync_func(delay=0.2):
            time.sleep(delay)
            return delay
        
        # Add independent tasks that should run concurrently
        dag.add_node("async1", lambda: asyncio.run(slow_async_func(0.2)))
        dag.add_node("async2", lambda: asyncio.run(slow_async_func(0.2)))
        dag.add_node("sync1", lambda: slow_sync_func(0.2))
        dag.add_node("sync2", lambda: slow_sync_func(0.2))
        
        start_time = time.time()
        results = await dag.execute_async()
        end_time = time.time()
        
        # Should take around 0.2 seconds (concurrent) rather than 0.8 seconds (sequential)
        execution_time = end_time - start_time
        assert execution_time < 0.6  # Allow some overhead
        assert len(results) == 4


class TestNodeStatus:
    def test_node_execution_timing(self):
        dag = DAG("test_dag")
        
        def timed_func():
            time.sleep(0.1)
            return 42
        
        dag.add_node("timed_node", timed_func)
        
        results = dag.execute_sync()
        
        node = dag.nodes["timed_node"]
        assert node.execution_time >= 0.1
        assert node.status == NodeStatus.COMPLETED
        assert node.result == 42
        assert node.error is None

    def test_node_error_status(self):
        dag = DAG("test_dag")
        
        def failing_func():
            raise RuntimeError("Test error")
        
        dag.add_node("failing_node", failing_func)
        
        results = dag.execute_sync()
        
        node = dag.nodes["failing_node"]
        assert node.status == NodeStatus.FAILED
        assert node.result is None
        assert isinstance(node.error, RuntimeError)
        assert str(node.error) == "Test error"


class TestConditionalExecution:
    def test_node_with_true_condition(self):
        dag = DAG("test_dag")
        
        def always_true():
            return True
        
        def test_func():
            return 42
        
        dag.add_node("conditional_node", test_func, condition=always_true)
        
        results = dag.execute_sync()
        
        assert results["conditional_node"] == 42
        assert dag.nodes["conditional_node"].status == NodeStatus.COMPLETED

    def test_node_with_false_condition(self):
        dag = DAG("test_dag")
        
        def always_false():
            return False
        
        def test_func():
            return 42
        
        dag.add_node("conditional_node", test_func, condition=always_false)
        
        results = dag.execute_sync()
        
        assert "conditional_node" not in results
        assert dag.nodes["conditional_node"].status == NodeStatus.CONDITION_NOT_MET

    def test_node_with_condition_exception(self):
        dag = DAG("test_dag")
        
        def failing_condition():
            raise ValueError("Condition error")
        
        def test_func():
            return 42
        
        dag.add_node("conditional_node", test_func, condition=failing_condition)
        
        results = dag.execute_sync()
        
        assert "conditional_node" not in results
        assert dag.nodes["conditional_node"].status == NodeStatus.CONDITION_NOT_MET

    def test_conditional_dependencies(self):
        dag = DAG("test_dag")
        
        def always_true():
            return True
        
        def always_false():
            return False
        
        def func1():
            return 10
        
        def func2():
            return 20
        
        def dependent_func():
            # This should not execute because conditional_node won't run
            return 30
        
        dag.add_node("node1", func1)
        dag.add_node("conditional_node", func2, dependencies=["node1"], condition=always_false)
        dag.add_node("dependent_node", dependent_func, dependencies=["conditional_node"])
        
        results = dag.execute_sync()
        
        assert results["node1"] == 10
        assert "conditional_node" not in results
        assert "dependent_node" not in results
        assert dag.nodes["conditional_node"].status == NodeStatus.CONDITION_NOT_MET
        assert dag.nodes["dependent_node"].status == NodeStatus.SKIPPED

    def test_conditional_task_decorator(self):
        dag = DAG("test_dag")
        
        def condition_check():
            return True
        
        @task(dag, "conditional_task", condition=condition_check)
        def conditional_task():
            return 42
        
        results = dag.execute_sync()
        
        assert results["conditional_task"] == 42
        assert dag.nodes["conditional_task"].status == NodeStatus.COMPLETED

    def test_conditional_task_decorator_false(self):
        dag = DAG("test_dag")
        
        def condition_check():
            return False
        
        @task(dag, "conditional_task", condition=condition_check)
        def conditional_task():
            return 42
        
        results = dag.execute_sync()
        
        assert "conditional_task" not in results
        assert dag.nodes["conditional_task"].status == NodeStatus.CONDITION_NOT_MET

    @pytest.mark.asyncio
    async def test_conditional_async_execution(self):
        dag = DAG("test_dag")
        
        def condition_true():
            return True
        
        def condition_false():
            return False
        
        async def async_func1():
            await asyncio.sleep(0.1)
            return 10
        
        async def async_func2():
            await asyncio.sleep(0.1)
            return 20
        
        dag.add_node("async_node1", async_func1, condition=condition_true)
        dag.add_node("async_node2", async_func2, condition=condition_false)
        
        results = await dag.execute_async()
        
        assert results["async_node1"] == 10
        assert "async_node2" not in results
        assert dag.nodes["async_node1"].status == NodeStatus.COMPLETED
        assert dag.nodes["async_node2"].status == NodeStatus.CONDITION_NOT_MET

    def test_dynamic_condition_with_dag_state(self):
        dag = DAG("test_dag")
        
        def func1():
            return 5
        
        def condition_based_on_result():
            # Condition depends on result of previous node
            return dag.nodes["node1"].result > 3
        
        def func2():
            return 20
        
        dag.add_node("node1", func1)
        dag.add_node("conditional_node", func2, dependencies=["node1"], condition=condition_based_on_result)
        
        results = dag.execute_sync()
        
        assert results["node1"] == 5
        assert results["conditional_node"] == 20
        assert dag.nodes["conditional_node"].status == NodeStatus.COMPLETED

    def test_dynamic_condition_false_with_dag_state(self):
        dag = DAG("test_dag")
        
        def func1():
            return 2
        
        def condition_based_on_result():
            # Condition depends on result of previous node
            return dag.nodes["node1"].result > 3
        
        def func2():
            return 20
        
        dag.add_node("node1", func1)
        dag.add_node("conditional_node", func2, dependencies=["node1"], condition=condition_based_on_result)
        
        results = dag.execute_sync()
        
        assert results["node1"] == 2
        assert "conditional_node" not in results
        assert dag.nodes["conditional_node"].status == NodeStatus.CONDITION_NOT_MET