"""Main DAG class for PyDAG."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import networkx as nx

from .core.node import Node, NodeStatus
from .core.retry import RetryConfig
from .exceptions import (
    CyclicGraphError, 
    DuplicateNodeError, 
    NodeNotFoundError
)
from .visualization import visualize_dag


class DAG:
    """Directed Acyclic Graph for executing functions with dependencies."""
    
    def __init__(self, name: str, max_workers: Optional[int] = None):
        """
        Initialize a new DAG.
        
        Args:
            name: Name of the DAG
            max_workers: Maximum number of concurrent workers (defaults to CPU count)
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.max_workers = max_workers
        self._graph = nx.DiGraph()
        self.logger = logging.getLogger(f"dagexec.{name}")
    
    def add_node(self, name: str, func: Callable, dependencies: Optional[List[str]] = None, condition: Optional[Callable[[], bool]] = None, retry_config: Optional[RetryConfig] = None) -> Node:
        """
        Add a function as a node to the DAG.
        
        Args:
            name: Unique name for the node
            func: Function to execute
            dependencies: List of node names this node depends on
            condition: Optional condition function that must return True for the node to execute
            retry_config: Optional retry configuration for failed executions
            
        Returns:
            The created Node object
            
        Raises:
            DuplicateNodeError: If node with same name already exists
            NodeNotFoundError: If dependency does not exist
        """
        if name in self.nodes:
            raise DuplicateNodeError(f"Node '{name}' already exists in the DAG")
        
        deps = set(dependencies or [])
        
        # Verify dependencies exist
        for dep in deps:
            if dep not in self.nodes:
                raise NodeNotFoundError(f"Dependency '{dep}' does not exist in the DAG")
        
        node = Node(name=name, func=func, dependencies=deps, condition=condition, retry_config=retry_config)
        self.nodes[name] = node
        
        # Add to graph
        self._graph.add_node(name)
        for dep in deps:
            self._graph.add_edge(dep, name)
        
        return node
    
    def visualize(self, filename: Optional[str] = None) -> None:
        """
        Visualize the DAG.
        
        Args:
            filename: If provided, save the visualization to this file
        """
        visualize_dag(self._graph, self.nodes, filename)
    
    def verify_acyclic(self) -> bool:
        """
        Verify that the graph is acyclic.
        
        Returns:
            True if the graph is acyclic, False otherwise
        """
        try:
            cycles = list(nx.simple_cycles(self._graph))
            if cycles:
                self.logger.error(f"Cycles detected in DAG: {cycles}")
                return False
            return True
        except nx.NetworkXNoCycle:
            return True
    
    def identify_parallel_branches(self) -> List[List[str]]:
        """
        Identify independent parallel branches in the DAG.
        
        A parallel branch is a maximal set of nodes that can execute
        independently from other branches until they reconverge.
        
        Returns:
            List of branches, where each branch is a list of node names
        """
        if not self._graph.nodes:
            return []
        
        # Find root nodes (nodes with no predecessors)
        root_nodes = [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]
        
        # If only one root, we need to find where branches split
        if len(root_nodes) == 1:
            return self._find_branches_from_splits()
        
        # Multiple roots - each root starts its own branch
        branches = []
        for root in root_nodes:
            branch = self._get_branch_from_node(root)
            if branch:
                branches.append(branch)
        
        return branches
    
    def _find_branches_from_splits(self) -> List[List[str]]:
        """Find branches starting from split points in the DAG."""
        branches = []
        
        # Find nodes that have multiple successors (split points)
        split_nodes = [n for n in self._graph.nodes() if self._graph.out_degree(n) > 1]
        
        for split_node in split_nodes:
            successors = list(self._graph.successors(split_node))
            
            # Each successor path becomes a potential branch
            for successor in successors:
                branch = self._get_branch_from_node(successor, exclude_splits=True)
                if len(branch) > 1:  # Only consider non-trivial branches
                    branches.append(branch)
        
        # If no splits found, treat the entire DAG as one branch
        if not branches:
            all_nodes = list(nx.topological_sort(self._graph))
            return [all_nodes] if all_nodes else []
        
        return branches
    
    def _get_branch_from_node(self, start_node: str, exclude_splits: bool = False) -> List[str]:
        """
        Get all nodes reachable from a starting node until reconvergence.
        
        Args:
            start_node: Node to start the branch from
            exclude_splits: If True, stop at nodes with multiple successors
            
        Returns:
            List of node names in the branch
        """
        branch = [start_node]
        current = start_node
        visited = {start_node}
        
        while True:
            successors = [s for s in self._graph.successors(current) if s not in visited]
            
            if not successors:
                break
                
            if len(successors) > 1:
                if exclude_splits:
                    break
                # Multiple successors - this branch splits further
                # Include all reachable nodes
                for successor in successors:
                    sub_branch = self._get_branch_from_node(successor, exclude_splits=True)
                    for node in sub_branch:
                        if node not in visited:
                            branch.append(node)
                            visited.add(node)
                break
            
            # Single successor - continue the branch
            successor = successors[0]
            
            # Check if this successor has multiple predecessors (reconvergence point)
            if self._graph.in_degree(successor) > 1:
                # This is where branches reconverge - include it and stop
                branch.append(successor)
                break
                
            branch.append(successor)
            visited.add(successor)
            current = successor
        
        return branch
    
    def get_branch_execution_order(self, branch_nodes: List[str]) -> List[str]:
        """
        Get the execution order for nodes within a specific branch.
        
        Args:
            branch_nodes: List of node names in the branch
            
        Returns:
            Topologically sorted list of node names
        """
        # Create subgraph for this branch
        subgraph = self._graph.subgraph(branch_nodes)
        
        # Return topological sort of the subgraph
        return list(nx.topological_sort(subgraph))
    
    def get_ready_nodes(self) -> List[Node]:
        """
        Get nodes that are ready to execute (all dependencies are completed and conditions are met).
        
        Returns:
            List of nodes ready to execute
        """
        ready_nodes = []
        for name, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
                
            deps_satisfied = all(
                self.nodes[dep].status == NodeStatus.COMPLETED 
                for dep in node.dependencies
            )
            
            if not deps_satisfied:
                continue
            
            # Check condition if it exists
            if node.condition is not None:
                try:
                    condition_met = node.condition()
                    if not condition_met:
                        self.logger.info(f"Condition not met for '{node.name}', skipping")
                        node.status = NodeStatus.CONDITION_NOT_MET
                        continue
                except Exception as e:
                    self.logger.error(f"Error evaluating condition for '{node.name}': {str(e)}")
                    node.status = NodeStatus.CONDITION_NOT_MET
                    continue
            
            ready_nodes.append(node)
                
        return ready_nodes
    
    def execute_sync(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the DAG synchronously.
        
        Args:
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary mapping node names to their results
            
        Raises:
            CyclicGraphError: If the graph contains cycles
        """
        if not self.verify_acyclic():
            raise CyclicGraphError("Cannot execute cyclic graph")
        
        # Get topological sort of the nodes
        for node_name in nx.topological_sort(self._graph):
            node = self.nodes[node_name]
            
            # Check if any dependencies failed or conditions not met
            if any(self.nodes[dep].status in (NodeStatus.FAILED, NodeStatus.CONDITION_NOT_MET) for dep in node.dependencies):
                self.logger.warning(f"Skipping '{node.name}' due to failed or conditional dependencies")
                node.status = NodeStatus.SKIPPED
                continue
            
            # Check condition if it exists
            if node.condition is not None:
                try:
                    condition_met = node.condition()
                    if not condition_met:
                        self.logger.info(f"Condition not met for '{node.name}', skipping")
                        node.status = NodeStatus.CONDITION_NOT_MET
                        continue
                except Exception as e:
                    self.logger.error(f"Error evaluating condition for '{node.name}': {str(e)}")
                    node.status = NodeStatus.CONDITION_NOT_MET
                    continue
            
            try:
                self.logger.info(f"Executing '{node.name}'")
                if node.is_async:
                    # For async functions, run them in a new event loop
                    asyncio.run(node.func(**kwargs))
                else:
                    node.func(**kwargs)
                self.logger.info(f"Completed '{node.name}' in {node.execution_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to execute '{node.name}': {str(e)}")
                
        return {name: node.result for name, node in self.nodes.items() 
                if node.status == NodeStatus.COMPLETED}
    
    async def execute_async(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the DAG asynchronously with concurrency.
        
        Args:
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary mapping node names to their results
            
        Raises:
            CyclicGraphError: If the graph contains cycles
        """
        if not self.verify_acyclic():
            raise CyclicGraphError("Cannot execute cyclic graph")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.get_event_loop()
        
        # Keep track of completed nodes
        completed_nodes = set()
        pending_nodes = set(self.nodes.keys())
        
        while pending_nodes:
            # Get nodes ready to execute
            ready_nodes = self.get_ready_nodes()
            
            if not ready_nodes and pending_nodes:
                # Check if we're stuck due to failed dependencies
                stuck_nodes = [
                    node for name, node in self.nodes.items()
                    if node.status == NodeStatus.PENDING and 
                    any(self.nodes[dep].status in (NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CONDITION_NOT_MET) 
                        for dep in node.dependencies)
                ]
                
                for node in stuck_nodes:
                    self.logger.warning(f"Skipping '{node.name}' due to failed dependencies")
                    node.status = NodeStatus.SKIPPED
                    pending_nodes.remove(node.name)
                
                # Also remove nodes that have condition not met
                condition_not_met_nodes = [
                    name for name, node in self.nodes.items()
                    if node.status == NodeStatus.CONDITION_NOT_MET and name in pending_nodes
                ]
                
                for node_name in condition_not_met_nodes:
                    pending_nodes.remove(node_name)
                
                # If no nodes are ready and none were skipped, we may have a cycle
                if not stuck_nodes and not condition_not_met_nodes:
                    remaining = [name for name in pending_nodes]
                    raise RuntimeError(f"DAG execution stalled with pending nodes: {remaining}")
                
                continue
            
            # Execute ready nodes concurrently
            futures = []
            for node in ready_nodes:
                self.logger.info(f"Scheduling '{node.name}' for execution")
                if node.is_async:
                    # For async functions, create a task directly
                    future = asyncio.create_task(node.func(**kwargs))
                else:
                    # For sync functions, use executor
                    future = loop.run_in_executor(
                        executor, 
                        lambda n=node: n.func(**kwargs)
                    )
                futures.append((node, future))
            
            # Wait for executions to complete
            for node, future in futures:
                try:
                    await future
                    self.logger.info(f"Completed '{node.name}' in {node.execution_time:.2f}s")
                    completed_nodes.add(node.name)
                except Exception as e:
                    self.logger.error(f"Failed to execute '{node.name}': {str(e)}")
                    # Already handled by the wrapped function
                finally:
                    pending_nodes.discard(node.name)
        
        executor.shutdown()
        return {name: node.result for name, node in self.nodes.items() 
                if node.status == NodeStatus.COMPLETED}
    
    async def execute_parallel_branches(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the DAG with parallel branch optimization.
        
        This method identifies independent branches in the DAG and executes
        them concurrently, which can provide better performance than regular
        async execution when the DAG has distinct parallel paths.
        
        Args:
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary mapping node names to their results
            
        Raises:
            CyclicGraphError: If the graph contains cycles
        """
        if not self.verify_acyclic():
            raise CyclicGraphError("Cannot execute cyclic graph")
        
        # Identify parallel branches
        branches = self.identify_parallel_branches()
        
        if len(branches) <= 1:
            # No parallel branches found, fall back to regular async execution
            self.logger.info("No parallel branches detected, using regular async execution")
            return await self.execute_async(**kwargs)
        
        self.logger.info(f"Identified {len(branches)} parallel branches")
        for i, branch in enumerate(branches):
            self.logger.debug(f"Branch {i+1}: {branch}")
        
        # Execute branches concurrently
        branch_tasks = []
        for i, branch in enumerate(branches):
            task = asyncio.create_task(
                self._execute_branch(branch, f"branch_{i+1}", **kwargs)
            )
            branch_tasks.append(task)
        
        # Wait for all branches to complete
        branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)
        
        # Collect results from all branches
        all_results = {}
        for i, result in enumerate(branch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Branch {i+1} failed: {result}")
                # Continue with other branches
            elif isinstance(result, dict):
                all_results.update(result)
        
        # Execute any remaining nodes that weren't part of branches
        executed_nodes = set()
        for branch in branches:
            executed_nodes.update(branch)
        
        remaining_nodes = set(self.nodes.keys()) - executed_nodes
        if remaining_nodes:
            self.logger.info(f"Executing {len(remaining_nodes)} remaining nodes")
            remaining_results = await self._execute_remaining_nodes(remaining_nodes, **kwargs)
            all_results.update(remaining_results)
        
        return all_results
    
    async def _execute_branch(self, branch_nodes: List[str], branch_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a single branch of nodes.
        
        Args:
            branch_nodes: List of node names in this branch
            branch_name: Name for logging purposes
            **kwargs: Keyword arguments to pass to functions
            
        Returns:
            Dictionary mapping node names to their results
        """
        self.logger.info(f"Starting execution of {branch_name} with {len(branch_nodes)} nodes")
        
        # Get execution order for this branch
        execution_order = self.get_branch_execution_order(branch_nodes)
        
        # Create a separate executor for this branch
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        loop = asyncio.get_event_loop()
        
        try:
            # Execute nodes in this branch following dependencies
            pending_nodes = set(execution_order)
            branch_results = {}
            
            while pending_nodes:
                # Get nodes ready to execute within this branch
                ready_nodes = []
                for node_name in pending_nodes:
                    node = self.nodes[node_name]
                    
                    if node.status != NodeStatus.PENDING:
                        continue
                    
                    # Check if dependencies are satisfied (within this branch or already completed)
                    deps_satisfied = all(
                        dep not in pending_nodes or self.nodes[dep].status == NodeStatus.COMPLETED
                        for dep in node.dependencies
                    )
                    
                    if not deps_satisfied:
                        continue
                    
                    # Check condition if it exists
                    if node.condition is not None:
                        try:
                            condition_met = node.condition()
                            if not condition_met:
                                self.logger.info(f"Condition not met for '{node.name}' in {branch_name}, skipping")
                                node.status = NodeStatus.CONDITION_NOT_MET
                                pending_nodes.remove(node_name)
                                continue
                        except Exception as e:
                            self.logger.error(f"Error evaluating condition for '{node.name}' in {branch_name}: {str(e)}")
                            node.status = NodeStatus.CONDITION_NOT_MET
                            pending_nodes.remove(node_name)
                            continue
                    
                    ready_nodes.append(node)
                
                if not ready_nodes and pending_nodes:
                    # Handle stuck nodes due to failed dependencies
                    stuck_nodes = [
                        node for node_name in pending_nodes
                        for node in [self.nodes[node_name]]
                        if node.status == NodeStatus.PENDING and 
                        any(self.nodes[dep].status in (NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CONDITION_NOT_MET) 
                            for dep in node.dependencies)
                    ]
                    
                    for node in stuck_nodes:
                        self.logger.warning(f"Skipping '{node.name}' in {branch_name} due to failed dependencies")
                        node.status = NodeStatus.SKIPPED
                        pending_nodes.remove(node.name)
                    
                    if not stuck_nodes:
                        remaining = list(pending_nodes)
                        raise RuntimeError(f"{branch_name} execution stalled with pending nodes: {remaining}")
                    
                    continue
                
                # Execute ready nodes concurrently within this branch
                futures = []
                for node in ready_nodes:
                    self.logger.info(f"Scheduling '{node.name}' in {branch_name}")
                    if node.is_async:
                        future = asyncio.create_task(node.func(**kwargs))
                    else:
                        future = loop.run_in_executor(
                            executor,
                            lambda n=node: n.func(**kwargs)
                        )
                    futures.append((node, future))
                
                # Wait for executions to complete
                for node, future in futures:
                    try:
                        await future
                        self.logger.info(f"Completed '{node.name}' in {branch_name} in {node.execution_time:.2f}s")
                        branch_results[node.name] = node.result
                    except Exception as e:
                        self.logger.error(f"Failed to execute '{node.name}' in {branch_name}: {str(e)}")
                    finally:
                        pending_nodes.discard(node.name)
            
            self.logger.info(f"Completed {branch_name} with {len(branch_results)} successful nodes")
            return branch_results
            
        finally:
            executor.shutdown()
    
    async def _execute_remaining_nodes(self, remaining_nodes: set, **kwargs) -> Dict[str, Any]:
        """Execute nodes that weren't part of any parallel branch."""
        results = {}
        
        # Simple topological execution for remaining nodes
        remaining_graph = self._graph.subgraph(remaining_nodes)
        execution_order = list(nx.topological_sort(remaining_graph))
        
        for node_name in execution_order:
            node = self.nodes[node_name]
            
            # Check dependencies
            deps_satisfied = all(
                self.nodes[dep].status == NodeStatus.COMPLETED
                for dep in node.dependencies
            )
            
            if not deps_satisfied:
                self.logger.warning(f"Skipping '{node.name}' due to unsatisfied dependencies")
                node.status = NodeStatus.SKIPPED
                continue
            
            # Check condition
            if node.condition is not None:
                try:
                    condition_met = node.condition()
                    if not condition_met:
                        self.logger.info(f"Condition not met for '{node.name}', skipping")
                        node.status = NodeStatus.CONDITION_NOT_MET
                        continue
                except Exception as e:
                    self.logger.error(f"Error evaluating condition for '{node.name}': {str(e)}")
                    node.status = NodeStatus.CONDITION_NOT_MET
                    continue
            
            # Execute the node
            try:
                self.logger.info(f"Executing remaining node '{node.name}'")
                if node.is_async:
                    await node.func(**kwargs)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: node.func(**kwargs))
                
                results[node.name] = node.result
                self.logger.info(f"Completed '{node.name}' in {node.execution_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to execute '{node.name}': {str(e)}")
        
        return results
    
    def execute(self, async_execution: bool = True, parallel_branches: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Execute the DAG.
        
        Args:
            async_execution: Whether to execute asynchronously with concurrency
            parallel_branches: Whether to use parallel branch optimization
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary mapping node names to their results
        """
        execution_mode = "parallel branches" if parallel_branches else ("async" if async_execution else "sync")
        self.logger.info(f"Executing DAG '{self.name}' with {len(self.nodes)} nodes using {execution_mode} mode")
        
        start_time = time.time()
        
        if parallel_branches:
            # Use parallel branch execution
            try:
                loop = asyncio.get_running_loop()
                # We're already in an async context, run directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.execute_parallel_branches(**kwargs))
                    )
                    results = future.result()
            except RuntimeError:
                # No running loop, create one
                results = asyncio.run(self.execute_parallel_branches(**kwargs))
        elif async_execution:
            try:
                loop = asyncio.get_running_loop()
                # We're already in an async context, run directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.execute_async(**kwargs))
                    )
                    results = future.result()
            except RuntimeError:
                # No running loop, create one
                results = asyncio.run(self.execute_async(**kwargs))
        else:
            results = self.execute_sync(**kwargs)
            
        total_time = time.time() - start_time
        
        successful = sum(1 for node in self.nodes.values() 
                         if node.status == NodeStatus.COMPLETED)
        failed = sum(1 for node in self.nodes.values() 
                     if node.status == NodeStatus.FAILED)
        skipped = sum(1 for node in self.nodes.values() 
                      if node.status == NodeStatus.SKIPPED)
        
        self.logger.info(
            f"DAG execution completed in {total_time:.2f}s: "
            f"{successful} succeeded, {failed} failed, {skipped} skipped"
        )
        
        return results