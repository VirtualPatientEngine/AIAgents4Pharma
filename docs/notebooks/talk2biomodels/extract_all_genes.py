import os
import numpy as np
import pandas as pd
from aiagents4pharma.talk2knowledgegraphs.datasets.primekg import PrimeKG

def extract_genes_from_go(go_keywords=['drug discovery', 'immune system process', 'inflammatory response']):
    """
    Extract genes/proteins that are associated with GO terms related to immune and inflammation processes.

    Parameters:
    - go_keywords (list): List of keywords to match GO terms (default: ["immune", "inflammation"]).

    Returns:
    - genes_df (pd.DataFrame): DataFrame containing genes/proteins linked to the selected GO terms.
    """

    # Load PrimeKG dataset
    primekg_data = PrimeKG(local_dir="../../../../data/primekg/")
    primekg_data.load_data()

    # Get all nodes and edges
    primekg_nodes = primekg_data.get_nodes()
    primekg_edges = primekg_data.get_edges()

    # 🔹 STEP 1: Extract Relevant GO Terms
    go_query = "|".join(go_keywords)  # Create OR search pattern
    go_terms_df = primekg_nodes[
        (primekg_nodes["node_type"].isin(["biological_process", "molecular_function", "cellular_component"])) &
        (primekg_nodes["node_name"].str.contains(go_query, case=False, na=False))
    ]

    if go_terms_df.empty:
        print(f"No GO terms matching {go_keywords} found in PrimeKG!")
        return None

    go_term_ids = go_terms_df.index.values
    print(f"Found {len(go_term_ids)} GO terms related to {go_keywords}.")

    # 🔹 STEP 2: Extract Gene-GO Relationships
    gene_go_edges_df = primekg_edges[
        ((primekg_edges.head_index.isin(go_term_ids)) & (primekg_edges.tail_type == "gene/protein")) |
        ((primekg_edges.tail_index.isin(go_term_ids)) & (primekg_edges.head_type == "gene/protein"))
    ]

    if gene_go_edges_df.empty:
        print(f"No gene-GO relationships found for {go_keywords}!")
        return None

    gene_ids = np.unique(
        np.concatenate([
            gene_go_edges_df[gene_go_edges_df.head_type == "gene/protein"].head_index.unique(),
            gene_go_edges_df[gene_go_edges_df.tail_type == "gene/protein"].tail_index.unique()
        ])
    )

    print(f"Found {len(gene_ids)} genes/proteins linked to immune/inflammation GO terms.")

    # 🔹 STEP 3: Extract the Gene Nodes
    genes_df = primekg_nodes[primekg_nodes.index.isin(gene_ids)]
    genes_df.to_csv('genes_df.csv')

    print(f"Extracted {len(genes_df)} gene/protein nodes from PrimeKG based on immune/inflammation GO terms.")

    return genes_df


genes_df = extract_genes_from_go()