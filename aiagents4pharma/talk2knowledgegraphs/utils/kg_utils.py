#!/usr/bin/env python3

'''A utility module for knowledge graph operations'''

from typing import Tuple
import networkx as nx
import pandas as pd

def kg_to_df_pandas(kg: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a directed knowledge graph to a pandas DataFrame.

    Args:
        kg: The directed knowledge graph in networkX format.

    Returns:
        df_nodes: A pandas DataFrame of the nodes in the knowledge graph.
        df_edges: A pandas DataFrame of the edges in the knowledge graph.
    """

    # Create a pandas DataFrame of the nodes
    df_nodes = pd.DataFrame.from_dict(kg.nodes, orient='index')

    # Create a pandas DataFrame of the edges
    df_edges = nx.to_pandas_edgelist(kg,
                                    source='node_source',
                                    target='node_target')

    return df_nodes, df_edges

def df_pandas_to_kg(df: pd.DataFrame,
                    df_nodes_attrs: pd.DataFrame,
                    node_source: str,
                    node_target: str
                    ) -> nx.DiGraph:
    """
    Convert a pandas DataFrame to a directed knowledge graph.

    Args:
        df: A pandas DataFrame of the edges in the knowledge graph.
        df_nodes_attrs: A pandas DataFrame of the nodes in the knowledge graph.
        node_source: The column name of the source node in the df.
        node_target: The column name of the target node in the df.

    Returns:
        kg: The directed knowledge graph in networkX format.
    """

    # Assert if the columns node_source and node_target are in the df
    assert node_source in df.columns, f'{node_source} not in df'
    assert node_target in df.columns, f'{node_target} not in df'

    # Assert that the nodes in the index of the df_nodes_attrs
    # are present in the source and target columns of the df
    assert set(df_nodes_attrs.index).issubset(set(df[node_source]).\
                                        union(set(df[node_target]))), \
                                        'Nodes in index of df_nodes not found in df_edges'

    # Create a knowledge graph from the dataframes
    # Add edges and nodes to the knowledge graph
    kg = nx.from_pandas_edgelist(df,
                                source=node_source,
                                target=node_target,
                                create_using=nx.DiGraph,
                                edge_attr=True)
    kg.add_nodes_from(df_nodes_attrs.to_dict('index').items())

    return kg
