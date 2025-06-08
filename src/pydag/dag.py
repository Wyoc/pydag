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
    
    def execute(self, async_execution: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute the DAG.
        
        Args:
            async_execution: Whether to execute asynchronously with concurrency
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary mapping node names to their results
        """
        self.logger.info(f"Executing DAG '{self.name}' with {len(self.nodes)} nodes")
        
        start_time = time.time()
        
        if async_execution:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            results = loop.run_until_complete(self.execute_async(**kwargs))
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