"""
Test cases for utils/kg_utils.py
"""

import pytest
import networkx as nx
import pandas as pd
from ..utils import kg_utils


@pytest.fixture(name="sample_graph")
def make_sample_graph():
    """Return a sample graph"""
    sg = nx.Graph()
    sg.add_node(1, node_id=1, feature_id="A", feature_value="ValueA")
    sg.add_node(2, node_id=2, feature_id="B", feature_value="ValueB")
    sg.add_edge(1, 2, edge_id=1, feature_id="E", feature_value="EdgeValue")
    return sg


def test_kg_to_df_pandas(sample_graph):
    """Test the kg_to_df_pandas function"""
    df_nodes, df_edges = kg_utils.kg_to_df_pandas(sample_graph)
    print(df_nodes)
    expected_nodes_data = {
        "node_id": [1, 2],
        "feature_id": ["A", "B"],
        "feature_value": ["ValueA", "ValueB"],
    }
    expected_nodes_df = pd.DataFrame(expected_nodes_data, index=[1, 2])
    print(expected_nodes_df)
    expected_edges_data = {
        "node_source": [1],
        "node_target": [2],
        "edge_id": [1],
        "feature_id": ["E"],
        "feature_value": ["EdgeValue"],
    }
    expected_edges_df = pd.DataFrame(expected_edges_data)

    # Assert that the dataframes are equal but the order of columns may be different
    # Ignore the index of the dataframes
    pd.testing.assert_frame_equal(df_nodes, expected_nodes_df, check_like=True)
    pd.testing.assert_frame_equal(df_edges, expected_edges_df, check_like=True)


def test_df_pandas_to_kg():
    """Test the df_pandas_to_kg function"""
    nodes_data = {
        "node_id": [1, 2],
        "feature_id": ["A", "B"],
        "feature_value": ["ValueA", "ValueB"],
    }
    df_nodes_attrs = pd.DataFrame(nodes_data).set_index("node_id")

    edges_data = {
        "node_source": [1],
        "node_target": [2],
        "edge_id": [1],
        "feature_id": ["E"],
        "feature_value": ["EdgeValue"],
    }
    df_edges = pd.DataFrame(edges_data)

    kg = kg_utils.df_pandas_to_kg(
        df_edges, df_nodes_attrs, "node_source", "node_target"
    )

    assert len(kg.nodes) == 2
    assert len(kg.edges) == 1

    assert kg.nodes[1]["feature_id"] == "A"
    assert kg.nodes[1]["feature_value"] == "ValueA"
    assert kg.nodes[2]["feature_id"] == "B"
    assert kg.nodes[2]["feature_value"] == "ValueB"

    assert kg.edges[1, 2]["feature_id"] == "E"
    assert kg.edges[1, 2]["feature_value"] == "EdgeValue"
    assert kg.edges[1, 2]["edge_id"] == 1
