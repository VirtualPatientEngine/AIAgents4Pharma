import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
from aiagents4pharma.talk2knowledgegraphs.datasets.primekg import PrimeKG

def extract_disease_subgraph(disease_names=["crohn", "inflammatory bowel disease", "ulcerative colitis"]):
    """
    Extracts the subgraph of genes, GO terms, and ontologies related to given diseases.

    Parameters:
    - disease_names (list): List of disease names (default includes Crohn's, IBD, and Ulcerative Colitis).

    Returns:
    - subgraph_nodes_df (pd.DataFrame): DataFrame containing the nodes in the extracted subgraph.
    - subgraph_edges_df (pd.DataFrame): DataFrame containing the edges in the extracted subgraph.
    """

    # Load PrimeKG dataset
    primekg_data = PrimeKG(local_dir="../../../../data/primekg/")
    primekg_data.load_data()

    # Get all nodes and edges
    primekg_nodes = primekg_data.get_nodes()
    primekg_edges = primekg_data.get_edges()

    # 🔹 STEP 1: Extract Disease Nodes
    disease_names = [d.lower() for d in disease_names]
    disease_query = "|".join(disease_names)  # Combine all disease names for OR search
    disease_nodes_df = primekg_nodes[
        (primekg_nodes["node_type"] == "disease") &
        (primekg_nodes["node_name"].str.contains(disease_query, case=False, na=False))
    ]

    if disease_nodes_df.empty:
        print(f"⚠️ No matching disease found in PrimeKG!")
        return None, None

    disease_ids = disease_nodes_df.index.values
    print(f"✅ Found {len(disease_ids)} disease nodes for '{disease_names}'.")

    # 🔹 STEP 2: Extract Disease-Gene Relationships
    disease_gene_edges_df = primekg_edges[
        ((primekg_edges.head_index.isin(disease_ids)) & (primekg_edges.tail_type == "gene/protein")) |
        ((primekg_edges.tail_index.isin(disease_ids)) & (primekg_edges.head_type == "gene/protein"))
    ]

    gene_ids = np.unique(
        np.concatenate([
            disease_gene_edges_df[disease_gene_edges_df.head_type == "gene/protein"].head_index.unique(),
            disease_gene_edges_df[disease_gene_edges_df.tail_type == "gene/protein"].tail_index.unique()
        ])
    )
    print(f"✅ Found {len(gene_ids)} genes/proteins related to '{disease_names}'.")

    # 🔹 STEP 3: Extract GO Terms (biological_process, molecular_function, cellular_component)
    go_terms_df = primekg_nodes[
        primekg_nodes["node_type"].isin(["biological_process", "molecular_function", "cellular_component"])
    ]
    go_term_ids = go_terms_df.index.values

    # 🔹 STEP 4: Extract Ontologies (SNOMEDCT, BTO, FMA, Anatomy)
    ontology_nodes = primekg_nodes[
        primekg_nodes["node_type"].isin(["SNOMEDCT", "BTO", "FMA", "anatomy"])
    ]
    ontology_ids = ontology_nodes.index.values

    # 🔹 STEP 5: Extract Subgraph Edges (Gene → GO Terms → Ontologies)
    subgraph_edges_df = primekg_edges[
        ((primekg_edges.head_index.isin(gene_ids)) & (primekg_edges.tail_index.isin(go_term_ids))) |
        ((primekg_edges.tail_index.isin(gene_ids)) & (primekg_edges.head_index.isin(go_term_ids))) |
        ((primekg_edges.head_index.isin(go_term_ids)) & (primekg_edges.tail_index.isin(ontology_ids))) |
        ((primekg_edges.tail_index.isin(go_term_ids)) & (primekg_edges.head_index.isin(ontology_ids))) |
        ((primekg_edges.head_index.isin(gene_ids)) & (primekg_edges.tail_index.isin(ontology_ids))) |
        ((primekg_edges.tail_index.isin(gene_ids)) & (primekg_edges.head_index.isin(ontology_ids)))
    ]

    # 🔹 STEP 6: Extract All Related Nodes
    subgraph_node_ids = np.unique(
        np.hstack([subgraph_edges_df.head_index.unique(), subgraph_edges_df.tail_index.unique()])
    )
    subgraph_nodes_df = primekg_nodes[primekg_nodes.index.isin(subgraph_node_ids)]

    print(f"✅ Final subgraph contains {len(subgraph_nodes_df)} nodes and {len(subgraph_edges_df)} edges.")

    return subgraph_nodes_df, subgraph_edges_df


# Example Usage
subgraph_nodes, subgraph_edges = extract_disease_subgraph(
    ["Crohn's Disease", "Inflammatory Bowel Disease", "Ulcerative Colitis"]
)

# To extract for another set of diseases, simply change the list:
# subgraph_nodes, subgraph_edges = extract_disease_subgraph(["Lung Cancer", "Breast Cancer"])
