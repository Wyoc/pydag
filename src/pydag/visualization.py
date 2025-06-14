"""Visualization utilities for PyDAG."""

from typing import Optional, Dict, TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from .core.node import Node, NodeStatus


def visualize_dag(graph: nx.DiGraph, nodes: Dict[str, "Node"], filename: Optional[str] = None) -> None:
    """Visualize the DAG structure and node execution statuses.
    
    Creates a visual representation of the DAG showing nodes as colored circles
    based on their execution status and edges representing dependencies.
    
    Args:
        graph: NetworkX DiGraph representing the DAG structure.
        nodes: Dictionary mapping node names to Node objects with status info.
        filename: Optional file path to save the visualization. If not provided,
            displays the plot interactively.
            
    Note:
        Node colors indicate status:
        - Gray: Pending
        - Blue: Running  
        - Green: Completed
        - Red: Failed
        - Orange: Skipped
        - Yellow: Condition not met
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    
    node_colors = []
    for node_name in graph.nodes():
        node = nodes[node_name]
        node_colors.append(_get_node_color(node.status))
    
    nx.draw(
        graph, pos, with_labels=True, node_color=node_colors,
        node_size=3000, font_size=10, font_weight='bold', arrows=True,
        connectionstyle='arc3, rad=0.1'
    )
    
    if filename:
        plt.savefig(filename)
    plt.show()


def _get_node_color(status: "NodeStatus") -> str:
    """Get color for a node based on its status."""
    from .core.node import NodeStatus
    
    color_map = {
        NodeStatus.COMPLETED: 'green',
        NodeStatus.RUNNING: 'yellow', 
        NodeStatus.FAILED: 'red',
        NodeStatus.SKIPPED: 'gray',
        NodeStatus.CONDITION_NOT_MET: 'orange',
        NodeStatus.PENDING: 'blue'
    }
    
    return color_map.get(status, 'blue')