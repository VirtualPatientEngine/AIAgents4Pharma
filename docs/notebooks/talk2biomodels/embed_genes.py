import os
import numpy as np
import pandas as pd
import openai
import faiss
import pickle  # To save/load embeddings
from dotenv import load_dotenv
from aiagents4pharma.talk2knowledgegraphs.datasets.primekg import PrimeKG

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDINGS_FILE = "gene_embeddings.pkl"  # File to store/reuse embeddings
INDEX_FILE = "faiss_index.bin"  # FAISS index storage
GENE_MAPPING_FILE = "gene_id_mapping.pkl"  # Mapping of FAISS index to gene details


def get_openai_embedding(text, model="text-embedding-3-large"):
    """Generate text embeddings using OpenAI's API."""
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def extract_genes_from_go(go_keywords=["immune", "inflammation"]):
    """Extract genes/proteins associated with GO terms (immune/inflammation)."""
    primekg_data = PrimeKG(local_dir="../../../../data/primekg/")
    primekg_data.load_data()

    primekg_nodes = primekg_data.get_nodes()
    primekg_edges = primekg_data.get_edges()

    # STEP 1: Extract GO Terms
    go_query = "|".join(go_keywords)
    go_terms_df = primekg_nodes[
        (primekg_nodes["node_type"].isin(["biological_process", "molecular_function", "cellular_component"])) &
        (primekg_nodes["node_name"].str.contains(go_query, case=False, na=False))
    ]
    
    if go_terms_df.empty:
        print(f"No GO terms matching {go_keywords} found in PrimeKG!")
        return None

    go_term_ids = go_terms_df.index.values
    print(f"Found {len(go_term_ids)} GO terms related to {go_keywords}.")

    # STEP 2: Extract Gene-GO Relationships
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
    
    genes_df = primekg_nodes[primekg_nodes.index.isin(gene_ids)][["node_index", "node_name", "node_id", "node_type"]]
    
    print(f"Extracted {len(genes_df)} gene/protein nodes from PrimeKG based on immune/inflammation GO terms.")
    return genes_df


def compute_embeddings_for_genes(genes_df):
    """Compute and save embeddings for genes, or load from cache if available."""
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading precomputed embeddings...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    print("🔹 Computing embeddings for all genes/proteins...")
    embeddings = {}
    
    for idx, row in genes_df.iterrows():
        try:
            embedding = get_openai_embedding(row["node_name"])
            embeddings[row["node_index"]] = embedding
        except Exception as e:
            print(f"Failed to embed: {row['node_name']} - {e}")

    # Save embeddings
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved {len(embeddings)} gene embeddings.")
    return embeddings


def build_faiss_index(genes_df, embeddings):
    """Build and save a FAISS index for fast similarity search."""
    if os.path.exists(INDEX_FILE) and os.path.exists(GENE_MAPPING_FILE):
        print("Loading FAISS index from file...")
        index = faiss.read_index(INDEX_FILE)
        with open(GENE_MAPPING_FILE, "rb") as f:
            gene_id_mapping = pickle.load(f)
        return index, gene_id_mapping

    gene_ids, vectors = zip(*embeddings.items())
    vectors = np.vstack(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    gene_id_mapping = {i: genes_df.loc[genes_df["node_index"] == gene_id, ["node_name", "node_id"]].values[0]
                       for i, gene_id in enumerate(gene_ids)}

    faiss.write_index(index, INDEX_FILE)
    with open(GENE_MAPPING_FILE, "wb") as f:
        pickle.dump(gene_id_mapping, f)

    print(f"FAISS index built and saved with {index.ntotal} vectors.")
    return index, gene_id_mapping


def find_closest_gene(description, index, gene_id_mapping):
    """Find the closest matching gene using FAISS."""
    if index is None or gene_id_mapping is None:
        print("FAISS index not built.")
        return None

    print("🔹 Embedding description with OpenAI API...")
    query_embedding = get_openai_embedding(description).astype("float32")

    print("🔹 Performing FAISS vector search...")
    distances, indices = index.search(query_embedding.reshape(1, -1), k=1)

    best_match_idx = indices[0][0]
    best_match_score = distances[0][0]
    best_match_gene, ncbi_id = gene_id_mapping[best_match_idx]

    print(f"Best match found: {best_match_gene} (NCBI ID: {ncbi_id}, Score: {best_match_score:.4f})")
    
    return {"Gene Name": best_match_gene, "NCBI ID": ncbi_id, "Score": best_match_score}




# Extract genes
genes_df = extract_genes_from_go()

# Compute or load embeddings
embeddings = compute_embeddings_for_genes(genes_df)

#  Build or load FAISS index
index, gene_id_mapping = build_faiss_index(genes_df, embeddings)

