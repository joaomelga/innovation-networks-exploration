"""
Network Analysis Module for Bipartite and General Network Analysis

This module provides functions for analyzing bipartite networks, computing projections,
structural metrics, nestedness, and visualization capabilities.
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict
import warnings

def create_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col=None, 
                          set_1_name='set_1', set_2_name='set_2'):
    """
    Create a bipartite graph from a DataFrame with two node sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing edges between two node sets
    node_set_1_col : str
        Column name for the first node set
    node_set_2_col : str
        Column name for the second node set
    weight_col : str, optional
        Column name for edge weights
    set_1_name : str
        Name for the first node set (for metadata)
    set_2_name : str
        Name for the second node set (for metadata)
    
    Returns:
    --------
    networkx.Graph
        Bipartite graph with node attributes indicating set membership
    """
    G = nx.Graph()
    
    # Get unique nodes for each set
    nodes_set_1 = set(df[node_set_1_col].dropna().unique())
    nodes_set_2 = set(df[node_set_2_col].dropna().unique())
    
    # Check for overlap (should be empty for true bipartite)
    overlap = nodes_set_1.intersection(nodes_set_2)
    if overlap:
        warnings.warn(f"Node sets overlap: {len(overlap)} common nodes. "
                     f"Graph may not be truly bipartite.")
    
    # Add nodes with bipartite attribute
    G.add_nodes_from(nodes_set_1, bipartite=0, node_set=set_1_name)
    G.add_nodes_from(nodes_set_2, bipartite=1, node_set=set_2_name)
    
    # Add edges
    for _, row in df.iterrows():
        node_1 = row[node_set_1_col]
        node_2 = row[node_set_2_col]
        
        if pd.notna(node_1) and pd.notna(node_2):
            if weight_col and weight_col in row and pd.notna(row[weight_col]):
                G.add_edge(node_1, node_2, weight=row[weight_col])
            else:
                G.add_edge(node_1, node_2)
    
    return G

def is_bipartite_graph(G):
    """
    Check if a graph is bipartite and return partition information.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to test
    
    Returns:
    --------
    dict
        Dictionary with 'is_bipartite', 'set_0', 'set_1' keys
    """
    is_bip = nx.is_bipartite(G)
    
    result = {'is_bipartite': is_bip}
    
    if is_bip:
        try:
            # Try to get sets from node attributes first
            set_0 = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
            set_1 = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 1}
            
            # If attributes don't exist, compute bipartition
            if not set_0 or not set_1:
                sets = bipartite.sets(G)
                set_0, set_1 = sets
            
            result['set_0'] = set_0
            result['set_1'] = set_1
        except:
            result['set_0'] = set()
            result['set_1'] = set()
    
    return result

def compute_bipartite_projections(G, weighted=True):
    """
    Compute projections of a bipartite graph onto both node sets.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph
    weighted : bool
        Whether to compute weighted projections
    
    Returns:
    --------
    dict
        Dictionary with 'projection_0', 'projection_1', and metadata
    """
    bip_info = is_bipartite_graph(G)
    
    if not bip_info['is_bipartite']:
        raise ValueError("Graph is not bipartite")
    
    set_0 = bip_info['set_0']
    set_1 = bip_info['set_1']
    
    if weighted:
        proj_0 = bipartite.weighted_projected_graph(G, set_0)
        proj_1 = bipartite.weighted_projected_graph(G, set_1)
    else:
        proj_0 = bipartite.projected_graph(G, set_0)
        proj_1 = bipartite.projected_graph(G, set_1)
    
    return {
        'projection_0': proj_0,
        'projection_1': proj_1,
        'set_0_size': len(set_0),
        'set_1_size': len(set_1),
        'set_0_name': list(G.nodes(data=True))[0][1].get('node_set', 'set_0') if G.nodes() else 'set_0',
        'set_1_name': list(G.nodes(data=True))[0][1].get('node_set', 'set_1') if G.nodes() else 'set_1'
    }

def compute_bipartite_metrics(G):
    """
    Compute various structural metrics for a bipartite graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph
    
    Returns:
    --------
    dict
        Dictionary containing various bipartite metrics
    """
    bip_info = is_bipartite_graph(G)
    
    if not bip_info['is_bipartite']:
        raise ValueError("Graph is not bipartite")
    
    set_0 = bip_info['set_0']
    set_1 = bip_info['set_1']
    
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['set_0_size'] = len(set_0)
    metrics['set_1_size'] = len(set_1)
    
    # Density
    metrics['density'] = bipartite.density(G, set_0)
    
    # Degree statistics
    degrees_0 = [G.degree(n) for n in set_0]
    degrees_1 = [G.degree(n) for n in set_1]
    
    metrics['avg_degree_set_0'] = np.mean(degrees_0) if degrees_0 else 0
    metrics['avg_degree_set_1'] = np.mean(degrees_1) if degrees_1 else 0
    metrics['max_degree_set_0'] = max(degrees_0) if degrees_0 else 0
    metrics['max_degree_set_1'] = max(degrees_1) if degrees_1 else 0
    
    # Clustering (redundancy)
    try:
        clustering = bipartite.clustering(G)
        metrics['avg_clustering'] = np.mean(list(clustering.values())) if clustering else 0
        metrics['clustering_set_0'] = np.mean([clustering[n] for n in set_0 if n in clustering])
        metrics['clustering_set_1'] = np.mean([clustering[n] for n in set_1 if n in clustering])
    except:
        metrics['avg_clustering'] = 0
        metrics['clustering_set_0'] = 0
        metrics['clustering_set_1'] = 0
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
    
    return metrics

def compute_nestedness_nodf(G):
    """
    Compute NODF (Nestedness based on Overlap and Decreasing Fill) metric.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph
    
    Returns:
    --------
    dict
        Dictionary with NODF score and component scores
    """
    bip_info = is_bipartite_graph(G)
    
    if not bip_info['is_bipartite']:
        raise ValueError("Graph is not bipartite")
    
    set_0 = sorted(bip_info['set_0'])
    set_1 = sorted(bip_info['set_1'])
    
    # Create adjacency matrix
    adj_matrix = np.zeros((len(set_0), len(set_1)))
    
    for i, node_0 in enumerate(set_0):
        for j, node_1 in enumerate(set_1):
            if G.has_edge(node_0, node_1):
                adj_matrix[i, j] = 1
    
    # Sort rows and columns by degree (decreasing)
    row_degrees = np.sum(adj_matrix, axis=1)
    col_degrees = np.sum(adj_matrix, axis=0)
    
    row_order = np.argsort(row_degrees)[::-1]
    col_order = np.argsort(col_degrees)[::-1]
    
    sorted_matrix = adj_matrix[row_order][:, col_order]
    
    # Compute NODF
    n_rows, n_cols = sorted_matrix.shape
    
    # Row-wise nestedness
    row_nestedness = 0
    row_pairs = 0
    
    for i in range(n_rows - 1):
        for j in range(i + 1, n_rows):
            if row_degrees[row_order[i]] > row_degrees[row_order[j]]:
                # Count overlap
                overlap = np.sum(sorted_matrix[i] * sorted_matrix[j])
                # Normalize by degree of less connected row
                if row_degrees[row_order[j]] > 0:
                    row_nestedness += overlap / row_degrees[row_order[j]]
                row_pairs += 1
    
    # Column-wise nestedness
    col_nestedness = 0
    col_pairs = 0
    
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            if col_degrees[col_order[i]] > col_degrees[col_order[j]]:
                # Count overlap
                overlap = np.sum(sorted_matrix[:, i] * sorted_matrix[:, j])
                # Normalize by degree of less connected column
                if col_degrees[col_order[j]] > 0:
                    col_nestedness += overlap / col_degrees[col_order[j]]
                col_pairs += 1
    
    # Average NODF
    row_nodf = (row_nestedness / row_pairs * 100) if row_pairs > 0 else 0
    col_nodf = (col_nestedness / col_pairs * 100) if col_pairs > 0 else 0
    total_nodf = (row_nodf + col_nodf) / 2
    
    return {
        'nodf_total': total_nodf,
        'nodf_rows': row_nodf,
        'nodf_cols': col_nodf,
        'adjacency_matrix': sorted_matrix,
        'row_order': row_order,
        'col_order': col_order
    }

def compute_cross_assortativity(G):
    """
    Compute cross-assortativity between the two sets in a bipartite graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph
    
    Returns:
    --------
    dict
        Dictionary with assortativity metrics
    """
    bip_info = is_bipartite_graph(G)
    
    if not bip_info['is_bipartite']:
        raise ValueError("Graph is not bipartite")
    
    set_0 = bip_info['set_0']
    set_1 = bip_info['set_1']
    
    # Get degrees for each set
    degrees_0 = {n: G.degree(n) for n in set_0}
    degrees_1 = {n: G.degree(n) for n in set_1}
    
    # For each edge, get the degrees of connected nodes
    edge_degrees_0 = []
    edge_degrees_1 = []
    
    for edge in G.edges():
        node_0, node_1 = edge
        if node_0 in set_0:
            edge_degrees_0.append(degrees_0[node_0])
            edge_degrees_1.append(degrees_1[node_1])
        else:
            edge_degrees_0.append(degrees_0[node_1])
            edge_degrees_1.append(degrees_1[node_0])
    
    # Compute correlation
    if len(edge_degrees_0) > 1:
        correlation, p_value = pearsonr(edge_degrees_0, edge_degrees_1)
    else:
        correlation, p_value = 0, 1
    
    return {
        'cross_assortativity': correlation,
        'p_value': p_value,
        'num_edges': len(edge_degrees_0)
    }

def analyze_bipartite_structure(df, node_set_1_col, node_set_2_col, weight_col=None,
                               set_1_name='set_1', set_2_name='set_2'):
    """
    Comprehensive analysis of bipartite network structure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing edges between two node sets
    node_set_1_col : str
        Column name for the first node set
    node_set_2_col : str
        Column name for the second node set
    weight_col : str, optional
        Column name for edge weights
    set_1_name : str
        Name for the first node set
    set_2_name : str
        Name for the second node set
    
    Returns:
    --------
    dict
        Comprehensive analysis results
    """
    # Create bipartite graph
    G = create_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col, 
                              set_1_name, set_2_name)
    
    results = {
        'graph': G,
        'basic_metrics': compute_bipartite_metrics(G),
        'nestedness': compute_nestedness_nodf(G),
        'cross_assortativity': compute_cross_assortativity(G),
        'projections': compute_bipartite_projections(G)
    }
    
    return results

def visualize_bipartite_graph(G, layout='spring', figsize=(12, 8), 
                             node_size_scale=1000, edge_alpha=0.5):
    """
    Visualize a bipartite graph with different colors for each set.
    
    Parameters:
    -----------
    G : networkx.Graph
        Bipartite graph to visualize
    layout : str
        Layout algorithm ('spring', 'bipartite', 'circular')
    figsize : tuple
        Figure size
    node_size_scale : int
        Base node size
    edge_alpha : float
        Edge transparency
    """
    bip_info = is_bipartite_graph(G)
    
    if not bip_info['is_bipartite']:
        print("Warning: Graph is not bipartite")
        return
    
    plt.figure(figsize=figsize)
    
    set_0 = bip_info['set_0']
    set_1 = bip_info['set_1']
    
    # Choose layout
    if layout == 'bipartite':
        pos = {}
        # Position set_0 on left, set_1 on right
        for i, node in enumerate(sorted(set_0)):
            pos[node] = (0, i / len(set_0))
        for i, node in enumerate(sorted(set_1)):
            pos[node] = (1, i / len(set_1))
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=set_0, 
                          node_color='lightblue', 
                          node_size=[G.degree(n) * node_size_scale for n in set_0],
                          alpha=0.8, label=f'Set 0 ({len(set_0)} nodes)')
    
    nx.draw_networkx_nodes(G, pos, nodelist=set_1, 
                          node_color='lightcoral',
                          node_size=[G.degree(n) * node_size_scale for n in set_1],
                          alpha=0.8, label=f'Set 1 ({len(set_1)} nodes)')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, edge_color='gray')
    
    plt.title(f'Bipartite Graph\n{len(set_0)} + {len(set_1)} nodes, {G.number_of_edges()} edges')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_nestedness_matrix(nestedness_result, figsize=(10, 8)):
    """
    Visualize the sorted adjacency matrix for nestedness analysis.
    
    Parameters:
    -----------
    nestedness_result : dict
        Result from compute_nestedness_nodf function
    figsize : tuple
        Figure size
    """
    matrix = nestedness_result['adjacency_matrix']
    
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap='Blues', cbar=True, 
                xticklabels=False, yticklabels=False)
    plt.title(f'Sorted Adjacency Matrix\nNODF Score: {nestedness_result["nodf_total"]:.2f}')
    plt.xlabel('Set 1 (sorted by degree)')
    plt.ylabel('Set 0 (sorted by degree)')
    plt.tight_layout()
    plt.show()

def compare_bipartite_networks(networks_dict):
    """
    Compare multiple bipartite networks using key metrics.
    
    Parameters:
    -----------
    networks_dict : dict
        Dictionary with network names as keys and analysis results as values
    
    Returns:
    --------
    pandas.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for name, analysis in networks_dict.items():
        metrics = analysis['basic_metrics']
        nestedness = analysis['nestedness']
        assortativity = analysis['cross_assortativity']
        
        comparison_data.append({
            'Network': name,
            'Nodes': metrics['num_nodes'],
            'Edges': metrics['num_edges'],
            'Density': metrics['density'],
            'Set_0_Size': metrics['set_0_size'],
            'Set_1_Size': metrics['set_1_size'],
            'Avg_Clustering': metrics['avg_clustering'],
            'Components': metrics['num_components'],
            'NODF_Score': nestedness['nodf_total'],
            'Cross_Assortativity': assortativity['cross_assortativity'],
            'Assortativity_p_value': assortativity['p_value']
        })
    
    return pd.DataFrame(comparison_data)

def extract_bipartite_edges_from_pairs(df, left_col, right_col, weight_col=None):
    """
    Extract unique bipartite edges from investment pairs data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with investment pairs
    left_col : str
        Column name for left node
    right_col : str
        Column name for right node
    weight_col : str, optional
        Column for edge weights
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with unique edges and aggregated weights
    """
    edges = df[[left_col, right_col]].copy()
    
    if weight_col:
        edges[weight_col] = df[weight_col]
        # Aggregate weights for duplicate edges
        edges = edges.groupby([left_col, right_col])[weight_col].sum().reset_index()
    else:
        # Just get unique edges
        edges = edges.drop_duplicates()
    
    return edges

def check_bipartite_overlap(df, node_set_1_col, node_set_2_col):
    """
    Check for overlapping nodes between two sets that should be bipartite.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing edges between two node sets
    node_set_1_col : str
        Column name for the first node set
    node_set_2_col : str
        Column name for the second node set
    
    Returns:
    --------
    dict
        Dictionary with overlap information
    """
    set_1 = set(df[node_set_1_col].dropna().unique())
    set_2 = set(df[node_set_2_col].dropna().unique())
    
    overlap = set_1.intersection(set_2)
    
    return {
        'set_1_size': len(set_1),
        'set_2_size': len(set_2),
        'overlap_size': len(overlap),
        'overlap_nodes': overlap,
        'is_truly_bipartite': len(overlap) == 0,
        'set_1_unique': set_1 - overlap,
        'set_2_unique': set_2 - overlap
    }

def create_pseudo_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col=None, 
                                 set_1_name='set_1', set_2_name='set_2', 
                                 handle_overlap='suffix'):
    """
    Create a bipartite-like graph by handling overlapping nodes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing edges between two node sets
    node_set_1_col : str
        Column name for the first node set
    node_set_2_col : str
        Column name for the second node set
    weight_col : str, optional
        Column name for edge weights
    set_1_name : str
        Name for the first node set
    set_2_name : str
        Name for the second node set
    handle_overlap : str
        How to handle overlapping nodes: 'suffix', 'remove', or 'warn'
    
    Returns:
    --------
    networkx.Graph
        Bipartite-like graph with handled overlaps
    """
    overlap_info = check_bipartite_overlap(df, node_set_1_col, node_set_2_col)
    
    if overlap_info['is_truly_bipartite']:
        # No overlap, use standard bipartite creation
        return create_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col, 
                                    set_1_name, set_2_name)
    
    print(f"Warning: {overlap_info['overlap_size']} overlapping nodes found between sets")
    print(f"Overlapping nodes: {list(overlap_info['overlap_nodes'])[:5]}{'...' if len(overlap_info['overlap_nodes']) > 5 else ''}")
    
    if handle_overlap == 'remove':
        # Remove edges involving overlapping nodes
        mask = ~df[node_set_1_col].isin(overlap_info['overlap_nodes']) & \
               ~df[node_set_2_col].isin(overlap_info['overlap_nodes'])
        df_clean = df[mask].copy()
        print(f"Removed {len(df) - len(df_clean)} edges involving overlapping nodes")
        return create_bipartite_graph(df_clean, node_set_1_col, node_set_2_col, weight_col, 
                                    set_1_name, set_2_name)
    
    elif handle_overlap == 'suffix':
        # Add suffixes to distinguish overlapping nodes
        df_modified = df.copy()
        df_modified[node_set_1_col] = df_modified[node_set_1_col].astype(str) + f'_{set_1_name}'
        df_modified[node_set_2_col] = df_modified[node_set_2_col].astype(str) + f'_{set_2_name}'
        print(f"Added suffixes to create pseudo-bipartite structure")
        return create_bipartite_graph(df_modified, node_set_1_col, node_set_2_col, weight_col, 
                                    set_1_name, set_2_name)
    
    else:  # warn
        warnings.warn(f"Graph has {overlap_info['overlap_size']} overlapping nodes and may not be truly bipartite")
        return create_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col, 
                                    set_1_name, set_2_name)

def analyze_bipartite_structure_robust(df, node_set_1_col, node_set_2_col, weight_col=None,
                                      set_1_name='set_1', set_2_name='set_2', 
                                      handle_overlap='suffix'):
    """
    Comprehensive analysis of bipartite network structure with overlap handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing edges between two node sets
    node_set_1_col : str
        Column name for the first node set
    node_set_2_col : str
        Column name for the second node set
    weight_col : str, optional
        Column name for edge weights
    set_1_name : str
        Name for the first node set
    set_2_name : str
        Name for the second node set
    handle_overlap : str
        How to handle overlapping nodes: 'suffix', 'remove', or 'warn'
    
    Returns:
    --------
    dict
        Comprehensive analysis results including overlap information
    """
    # Check for overlaps first
    overlap_info = check_bipartite_overlap(df, node_set_1_col, node_set_2_col)
    
    # Create bipartite graph with overlap handling
    G = create_pseudo_bipartite_graph(df, node_set_1_col, node_set_2_col, weight_col, 
                                    set_1_name, set_2_name, handle_overlap)
    
    results = {
        'graph': G,
        'overlap_info': overlap_info,
        'basic_metrics': compute_bipartite_metrics(G),
        'nestedness': compute_nestedness_nodf(G),
        'cross_assortativity': compute_cross_assortativity(G),
        'projections': compute_bipartite_projections(G)
    }
    
    return results