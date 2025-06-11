import os
import json
import numpy as np
import pandas as pd
import openai
import faiss
import pickle
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define file paths
DESCRIPTIONS_FILE = "descriptions_output.json"
EMBEDDINGS_FILE = "gene_embeddings.pkl"
INDEX_FILE = "faiss_index.bin"
GENE_MAPPING_FILE = "gene_id_mapping.pkl"
OUTPUT_FILE = "species_gene_matches.csv"


def get_openai_embedding(text, model="text-embedding-3-large"):
    """Generate text embeddings using OpenAI's API."""
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def load_faiss_index():
    """Load FAISS index and gene mapping."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(GENE_MAPPING_FILE):
        raise FileNotFoundError("FAISS index or gene mapping file not found. Run the gene extraction script first.")
    
    index = faiss.read_index(INDEX_FILE)
    with open(GENE_MAPPING_FILE, "rb") as f:
        gene_id_mapping = pickle.load(f)
    return index, gene_id_mapping


def search_species_genes(descriptions):
    """Embed species descriptions and find closest gene/protein match using FAISS default L2 distance."""
    index, gene_id_mapping = load_faiss_index()
    results = []
    
    for species, description in descriptions.items():
        query_embedding = get_openai_embedding(description).astype("float32")
        distances, indices = index.search(query_embedding.reshape(1, -1), k=1)
        best_match_idx = indices[0][0]
        best_match_score = distances[0][0]
        best_match_gene, ncbi_id = gene_id_mapping[best_match_idx]
        
        print(f"Best match for {species}: {best_match_gene} (NCBI ID: {ncbi_id}, Score: {best_match_score:.4f})")
        results.append({"species_name": species, "ncbi_node_id": ncbi_id})
    
    return results


def main():
    """Main function to read descriptions, search for genes, and save results."""
    if not os.path.exists(DESCRIPTIONS_FILE):
        raise FileNotFoundError("Descriptions JSON file not found!")
    
    with open(DESCRIPTIONS_FILE, "r") as f:
        descriptions = json.load(f)
    
    results = search_species_genes(descriptions)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()