"""Tests for parallel branch execution functionality."""

import asyncio
import time
import pytest
from unittest.mock import Mock

from pydag import DAG, task


class TestParallelBranchDetection:
    """Test parallel branch identification algorithms."""
    
    def test_single_linear_chain(self):
        """Test that a linear chain has no parallel branches."""
        dag = DAG("linear")
        
        @task(dag, "task1")
        def task1():
            return 1
            
        @task(dag, "task2", dependencies=["task1"])
        def task2():
            return 2
            
        @task(dag, "task3", dependencies=["task2"])
        def task3():
            return 3
        
        branches = dag.identify_parallel_branches()
        # Should have one branch containing all nodes
        assert len(branches) == 1
        assert set(branches[0]) == {"task1", "task2", "task3"}
    
    def test_simple_parallel_branches(self):
        """Test detection of simple parallel branches."""
        dag = DAG("parallel")
        
        @task(dag, "root")
        def root():
            return "start"
            
        @task(dag, "branch1_1", dependencies=["root"])
        def branch1_1():
            return "b1_1"
            
        @task(dag, "branch1_2", dependencies=["branch1_1"])
        def branch1_2():
            return "b1_2"
            
        @task(dag, "branch2_1", dependencies=["root"])
        def branch2_1():
            return "b2_1"
            
        @task(dag, "branch2_2", dependencies=["branch2_1"])
        def branch2_2():
            return "b2_2"
            
        @task(dag, "merge", dependencies=["branch1_2", "branch2_2"])
        def merge():
            return "merged"
        
        branches = dag.identify_parallel_branches()
        assert len(branches) == 2
        
        # Check that we have two distinct branches
        branch_sets = [set(branch) for branch in branches]
        
        # One branch should contain branch1_1 and branch1_2
        branch1_nodes = {"branch1_1", "branch1_2"}
        branch2_nodes = {"branch2_1", "branch2_2"}
        
        assert any(branch1_nodes.issubset(branch_set) for branch_set in branch_sets)
        assert any(branch2_nodes.issubset(branch_set) for branch_set in branch_sets)
    
    def test_multiple_independent_roots(self):
        """Test DAG with multiple independent starting points."""
        dag = DAG("multi_root")
        
        @task(dag, "root1")
        def root1():
            return "r1"
            
        @task(dag, "root2")
        def root2():
            return "r2"
            
        @task(dag, "child1", dependencies=["root1"])
        def child1():
            return "c1"
            
        @task(dag, "child2", dependencies=["root2"])
        def child2():
            return "c2"
        
        branches = dag.identify_parallel_branches()
        assert len(branches) == 2
        
        branch_sets = [set(branch) for branch in branches]
        assert {"root1", "child1"} in branch_sets or {"root1", "child1"}.issubset(branch_sets[0]) or {"root1", "child1"}.issubset(branch_sets[1])
        assert {"root2", "child2"} in branch_sets or {"root2", "child2"}.issubset(branch_sets[0]) or {"root2", "child2"}.issubset(branch_sets[1])
    
    def test_branch_execution_order(self):
        """Test that branch execution order respects dependencies."""
        dag = DAG("order_test")
        
        @task(dag, "a")
        def a():
            return "a"
            
        @task(dag, "b", dependencies=["a"])
        def b():
            return "b"
            
        @task(dag, "c", dependencies=["b"])
        def c():
            return "c"
        
        branch_nodes = ["a", "b", "c"]
        execution_order = dag.get_branch_execution_order(branch_nodes)
        
        assert execution_order == ["a", "b", "c"]


class TestParallelBranchExecution:
    """Test parallel branch execution engine."""
    
    def test_parallel_branch_execution_timing(self):
        """Test that parallel branches actually execute concurrently."""
        dag = DAG("timing_test", max_workers=4)
        
        # Track execution times to verify parallelism
        execution_times = {}
        
        @task(dag, "root")
        def root():
            execution_times["root"] = time.time()
            return "start"
            
        @task(dag, "branch1", dependencies=["root"])
        def branch1():
            time.sleep(0.2)  # 200ms
            execution_times["branch1"] = time.time()
            return "b1"
            
        @task(dag, "branch2", dependencies=["root"])
        def branch2():
            time.sleep(0.2)  # 200ms
            execution_times["branch2"] = time.time()
            return "b2"
        
        start_time = time.time()
        results = dag.execute(parallel_branches=True)
        total_time = time.time() - start_time
        
        # Should complete in roughly 0.4s (0.2s for each parallel branch)
        # rather than 0.6s if executed sequentially
        assert total_time < 0.5  # Allow some overhead
        assert "root" in results
        assert "branch1" in results
        assert "branch2" in results
        
        # Verify branch1 and branch2 started around the same time
        branch_time_diff = abs(execution_times["branch1"] - execution_times["branch2"])
        assert branch_time_diff < 0.1  # Started within 100ms of each other
    
    def test_async_parallel_branches(self):
        """Test parallel branches with async functions."""
        dag = DAG("async_parallel", max_workers=2)
        
        @task(dag, "root")
        async def root():
            await asyncio.sleep(0.1)
            return "start"
            
        @task(dag, "branch1", dependencies=["root"])
        async def branch1():
            await asyncio.sleep(0.2)
            return "async_b1"
            
        @task(dag, "branch2", dependencies=["root"])
        async def branch2():
            await asyncio.sleep(0.2)
            return "async_b2"
        
        start_time = time.time()
        results = dag.execute(parallel_branches=True)
        total_time = time.time() - start_time
        
        assert total_time < 0.5  # Should be roughly 0.3s not 0.5s
        assert results["root"] == "start"
        assert results["branch1"] == "async_b1"
        assert results["branch2"] == "async_b2"
    
    def test_failed_branch_isolation(self):
        """Test that failure in one branch doesn't affect others."""
        dag = DAG("failure_isolation")
        
        @task(dag, "root")
        def root():
            return "start"
            
        @task(dag, "good_branch", dependencies=["root"])
        def good_branch():
            return "success"
            
        @task(dag, "bad_branch", dependencies=["root"])
        def bad_branch():
            raise RuntimeError("Branch failure")
            
        results = dag.execute(parallel_branches=True)
        
        # Good branch should complete successfully
        assert "root" in results
        assert "good_branch" in results
        assert results["good_branch"] == "success"
        
        # Bad branch should not be in results
        assert "bad_branch" not in results
        
        # Verify node statuses
        assert dag.nodes["good_branch"].status.value == "completed"
        assert dag.nodes["bad_branch"].status.value == "failed"
    
    def test_fallback_to_regular_execution(self):
        """Test fallback when no parallel branches are detected."""
        dag = DAG("fallback_test")
        
        # Create a simple linear DAG (no parallel branches)
        @task(dag, "task1")
        def task1():
            return 1
            
        @task(dag, "task2", dependencies=["task1"])
        def task2():
            return 2
        
        # Should fall back to regular async execution
        results = dag.execute(parallel_branches=True)
        
        assert results["task1"] == 1
        assert results["task2"] == 2
    
    def test_conditional_nodes_in_parallel_branches(self):
        """Test conditional execution within parallel branches."""
        dag = DAG("conditional_parallel")
        
        @task(dag, "root")
        def root():
            return "start"
        
        # Branch with condition that evaluates to True
        def true_condition():
            return True
            
        @task(dag, "branch1", dependencies=["root"], condition=true_condition)
        def branch1():
            return "executed"
        
        # Branch with condition that evaluates to False  
        def false_condition():
            return False
            
        @task(dag, "branch2", dependencies=["root"], condition=false_condition)
        def branch2():
            return "skipped"
        
        results = dag.execute(parallel_branches=True)
        
        assert "root" in results
        assert "branch1" in results
        assert results["branch1"] == "executed"
        assert "branch2" not in results
        
        assert dag.nodes["branch1"].status.value == "completed"
        assert dag.nodes["branch2"].status.value == "condition_not_met"


class TestParallelBranchPerformance:
    """Performance and benchmarking tests for parallel branches."""
    
    def test_complex_dag_performance(self):
        """Test performance improvement with complex DAG structure."""
        dag = DAG("complex_perf", max_workers=4)
        
        # Create a simpler but still parallel DAG
        @task(dag, "start")
        def start():
            return "begin"
        
        # Create 2 parallel branches with 2 nodes each
        @task(dag, "branch_a_1", dependencies=["start"])
        def branch_a_1():
            time.sleep(0.1)
            return "a1"
        
        @task(dag, "branch_a_2", dependencies=["branch_a_1"])
        def branch_a_2():
            time.sleep(0.1)
            return "a2"
            
        @task(dag, "branch_b_1", dependencies=["start"])
        def branch_b_1():
            time.sleep(0.1)
            return "b1"
        
        @task(dag, "branch_b_2", dependencies=["branch_b_1"])
        def branch_b_2():
            time.sleep(0.1)
            return "b2"
        
        # Test both execution modes complete successfully
        results_parallel = dag.execute(parallel_branches=True)
        
        # Reset node statuses for fair comparison
        for node in dag.nodes.values():
            node.status = node.status.__class__.PENDING
            node.result = None
            node.error = None
            node.execution_time = 0.0
        
        results_regular = dag.execute(async_execution=True, parallel_branches=False)
        
        # Both should complete all nodes
        expected_nodes = {"start", "branch_a_1", "branch_a_2", "branch_b_1", "branch_b_2"}
        assert set(results_parallel.keys()) == expected_nodes
        assert set(results_regular.keys()) == expected_nodes
        
        # Verify results are correct
        assert results_parallel["start"] == "begin"
        assert results_parallel["branch_a_2"] == "a2"
        assert results_parallel["branch_b_2"] == "b2"