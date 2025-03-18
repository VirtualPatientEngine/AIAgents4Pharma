# Implement a Streamlit app to analyze a BioModel using a knowledge graph
import streamlit as st
import basico
import logging
import json
import torch
import pickle
import re
import pandas as pd
import sys
import os
import traceback
import requests
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add the project root to the path
# Get the current file's directory
current_dir = Path(__file__).resolve().parent
# Get the project root directory
project_root = current_dir
while project_root.name != "AIAgents4Pharma" and project_root != project_root.parent:
    project_root = project_root.parent

# Add the project root to the path
sys.path.insert(0, str(project_root))

# Now we can import from aiagents4pharma
from aiagents4pharma.talk2knowledgegraphs.utils.embeddings.ollama import (
    EmbeddingWithOllama,
)
from aiagents4pharma.talk2knowledgegraphs.datasets.primekg import PrimeKG

# Set file paths relative to the current directory
MODEL_PATH = current_dir / "Dwivedi_Model537_empty.xml"
PDF_PATH = current_dir / "pdf.pdf"
PYG_FILE = (
    project_root
    / "aiagents4pharma/talk2knowledgegraphs/tests/files/primekg_ibd_pyg_graph.pkl"
)
PRIMEKG_DIR = os.environ.get(
    "PRIMEKG_PATH", "/Users/anshkumar/Developer/code/git/data/primekg"
)

# Define compartment data
COMPARTMENTS_DATA = [
    {
        "compartment": "gut",
        "obo_id": "BTO:0000545",
        "description": "1: The alimentary canal or a portion thereof, especially the intestine or stomach. 2: The embryonic digestive tube, consisting of the foregut, the midgut, and the hindgut.",
    },
    {"compartment": "peripheral", "obo_id": None, "description": None},
    {
        "compartment": "liver",
        "obo_id": "BTO:0000759",
        "description": "1: A large very vascular glandular organ of vertebrates that secretes bile and causes important changes in many of the substances contained in the blood (as by converting sugars into glycogen which it stores up until required and by forming urea). 2: Any of various large compound glands associated with the digestive tract of invertebrate animals and probably concerned with the secretion of digestive enzymes.",
    },
    {
        "compartment": "serum",
        "obo_id": "BTO:0001239",
        "description": "1: The watery portion of an animal fluid remaining after coagulation: a (1): blood serum (2): antiserum b: whey c: a normal or pathological serous fluid (as in a blister). 2: The watery part of a plant fluid.",
    },
]

# Define UniProt ID mapping functions
API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 5


def submit_id_mapping(from_db, to_db, ids):
    """Submit a job to perform ID mapping."""
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    try:
        request.raise_for_status()
    except requests.HTTPError:
        st.error(f"Error submitting ID mapping job: {request.json()}")
        raise
    return request.json()["jobId"]


def check_id_mapping_results_ready(job_id):
    """Check if the ID mapping results are ready."""
    session = requests.Session()
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        try:
            request.raise_for_status()
        except requests.HTTPError:
            st.error(f"Error checking job status: {request.json()}")
            raise

        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] in ("NEW", "RUNNING"):
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_id_mapping_results_link(job_id):
    """Get the link to the ID mapping results."""
    """Get the link to the ID mapping results."""
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = requests.get(url)
    try:
        request.raise_for_status()
    except requests.HTTPError:
        st.error(f"Error getting results link: {request.json()}")
        raise
    return request.json()["redirectURL"]


def get_id_mapping_results_stream(url):
    """Get the ID mapping results from a stream."""
    import zlib
    from urllib.parse import urlparse, parse_qs

    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")

    session = requests.Session()
    request = session.get(url)
    try:
        request.raise_for_status()
    except requests.HTTPError:
        st.error(f"Error getting results: {request.json()}")
        raise

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )

    if compressed:
        decompressed = zlib.decompress(request.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            return json.loads(decompressed.decode("utf-8"))
    else:
        if file_format == "json":
            return request.json()

    return None


def extract_compartment_info(species_list):
    """Extract compartment information from species names."""
    # Define the regular expression pattern to match content between curly braces
    pattern = r"\{(.*?)\}"

    # Find all matches in the list of strings
    matches = set()
    for species in species_list:
        matches.update(re.findall(pattern, species))

    return list(matches)


def search_ols_term(ontology, term):
    """Search OLS for a term."""
    base_url = "https://www.ebi.ac.uk/ols/api/search"
    params = {"q": term, "ontology": ontology, "type": "class"}
    response = requests.get(
        base_url, params=params, headers={"Accept": "application/json"}, timeout=10
    )
    logger.info("OLS search response: %s", response.text)
    if response.status_code == 200:
        data = response.json()
        results = data.get("response", {}).get("docs", [])
        return results
    else:
        return f"Error: {response.status_code}"


def load_model_and_get_species(model_path):
    """Load SBML model and extract species list"""
    try:
        # Load the SBML model
        model = basico.load_model(str(model_path))
        # Get the model's species
        species = basico.get_species()
        species_list = species["display_name"].tolist()
        return species_list
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return []


def query_species_description(pdf_file_name, text_embedding_model, species_name):
    """
    Query PDF for species description using RAG
    """
    # Construct the retrieval query prompt
    retrieval_prompt = (
        f"Given the name of the species '{species_name}', please provide a concise description explaining its biological or functional significance. "
        "Your description should be clear, standalone, and should not reference the source article or include any citations."
    )
    logger.info("Retrieval Query Prompt: %s", retrieval_prompt)

    # Load the PDF pages
    loader = PyPDFLoader(str(pdf_file_name))
    pages = [page for page in loader.lazy_load()]
    logger.info("Loaded %d pages from the PDF", len(pages))

    # Create a vector store from the pages
    vector_store = InMemoryVectorStore.from_documents(pages, text_embedding_model)
    logger.info("Performing similarity search with the embedded query")

    # Retrieve documents based on the query
    docs = vector_store.similarity_search(retrieval_prompt)
    retrieved_text = "\n".join([doc.page_content for doc in docs])

    # Truncate text if too long
    max_chars = 3000
    if len(retrieved_text) > max_chars:
        retrieved_text = retrieved_text[:max_chars]
        logger.info("Retrieved text truncated to %d characters.", max_chars)

    # Define a prompt template for summarization
    summary_prompt = PromptTemplate(
        input_variables=["retrieved_text", "species_name"],
        template=(
            "You are provided with the following text excerpts:\n\n"
            "{retrieved_text}\n\n"
            "Based on this information, provide a concise, clear summary that explains what the species '{species_name}' means. "
            "Focus solely on the biological or functional description of the species and do not refer to the source text, article, or citations."
        ),
    )

    # Create a chain that uses the LLM and the prompt template
    chain = LLMChain(llm=OpenAI(temperature=0), prompt=summary_prompt)
    summary = chain.run(
        {"retrieved_text": retrieved_text, "species_name": species_name}
    )
    return summary


def generate_embeddings(species_descriptions):
    """Generate embeddings for species descriptions"""
    # Initialize embedding model
    emb_model = EmbeddingWithOllama(model_name="nomic-embed-text")

    # Generate embeddings for each description
    outputs = {
        key: emb_model.embed_documents([description])[0]
        for key, description in species_descriptions.items()
    }

    return outputs


def extract_protein_names(species_list):
    """Extract protein names from species list for UniProt querying."""
    pattern = r"([^{]+)"  # Match everything before the curly brace
    proteins = set()

    for species in species_list:
        match = re.match(pattern, species)
        if match:
            protein = match.group(1).strip()
            proteins.add(protein)

    return list(proteins)


def find_similar_nodes(embeddings, pyg_data):
    """Find similar nodes in knowledge graph using cosine similarity"""
    try:
        # Convert embeddings to tensor
        embeddings_tensor = torch.tensor([embeddings[key] for key in embeddings])
        st.write(f"Created embeddings tensor with shape {embeddings_tensor.shape}")

        # Get node features from PyG data (these are the embeddings)
        node_features = pyg_data.x
        st.write(f"Node features tensor has shape {node_features.shape}")

        # Calculate cosine similarity
        similarity = torch.nn.CosineSimilarity(dim=-1)(
            embeddings_tensor.unsqueeze(1),
            node_features.unsqueeze(0),
        )
        st.write(f"Calculated similarity with shape {similarity.shape}")

        # Find indices of maximum similarity
        max_indices = torch.argmax(similarity, dim=-1)
        st.write(f"Max similarity indices: {max_indices}")

        # Get node IDs - the data shows it has a node_id attribute
        node_ids = []
        for idx in max_indices:
            # Convert tensor index to integer
            idx_int = idx.item()
            # Get the node_id at this index
            node_id = pyg_data.node_id[idx_int]

            # If node_id is a tensor, convert to scalar
            if torch.is_tensor(node_id):
                node_id = node_id.item()

            # Extract numeric ID if needed
            try:
                # Check if node_id is a string with format like "something_(123)"
                if isinstance(node_id, str) and "(" in node_id and ")" in node_id:
                    numeric_id = int(re.sub("[()]", "", node_id.split("_")[-1]))
                    node_ids.append(numeric_id)
                else:
                    # Otherwise just use the ID as is
                    node_ids.append(node_id)
            except:
                # If any error occurs, use the node_id as is
                node_ids.append(node_id)

        st.write(f"Extracted node IDs: {node_ids}")
        return node_ids

    except Exception as e:
        st.error(f"Error in find_similar_nodes: {e}")
        st.code(traceback.format_exc())
        return []


def get_species_node_mapping(species_list, embeddings, pyg_data):
    """Get mapping between species and nodes using a different approach."""
    # Convert embeddings to tensors
    tensored_embeddings = {
        key: torch.tensor(embeddings[key]) for key in embeddings.keys()
    }

    # Calculate cosine similarity for each species individually
    computed_cosines = {}
    for key in tensored_embeddings.keys():
        # Make sure the node features are properly accessed
        if hasattr(pyg_data, "x"):
            node_features = pyg_data.x
            computed_cosines[key] = torch.nn.CosineSimilarity(dim=-1)(
                tensored_embeddings[key], node_features
            )
        else:
            st.error("PyG data doesn't have node features (x attribute)")
            return {}

    # Find best node index for each species
    best_nodes_index = {
        key: int(torch.argmax(computed_cosines[key])) for key in computed_cosines.keys()
    }

    # Get node IDs for each species
    specie_node = {}
    for key in best_nodes_index.keys():
        idx = best_nodes_index[key]
        if hasattr(pyg_data, "node_id"):
            node_id = pyg_data.node_id[idx]
            if torch.is_tensor(node_id):
                node_id = node_id.item()
            specie_node[key] = node_id

    # Extract numeric IDs
    node_id_mapping = {}
    for key in specie_node.keys():
        if (
            isinstance(specie_node[key], str)
            and "(" in specie_node[key]
            and ")" in specie_node[key]
        ):
            try:
                node_id_mapping[key] = int(
                    re.sub("[()]", "", specie_node[key].split("_")[-1])
                )
            except:
                node_id_mapping[key] = specie_node[key]
        else:
            node_id_mapping[key] = specie_node[key]

    return node_id_mapping


def main():
    """main function for Streamlit app"""
    st.title("BioModel Analysis with Knowledge Graph")

    st.sidebar.title("Settings")
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    if debug_mode:
        st.sidebar.write("Debug Information")
        st.sidebar.write(f"Current directory: {current_dir}")
        st.sidebar.write(f"Project root: {project_root}")
        st.sidebar.write(f"Model path: {MODEL_PATH}")
        st.sidebar.write(f"PDF path: {PDF_PATH}")
        st.sidebar.write(f"PyG file path: {PYG_FILE}")

    # Input files section
    st.header("Input Files")

    # Check if files exist
    model_exists = MODEL_PATH.exists()
    pdf_exists = PDF_PATH.exists()
    pyg_exists = PYG_FILE.exists()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Model: {'✅' if model_exists else '❌'}")
    with col2:
        st.write(f"PDF: {'✅' if pdf_exists else '❌'}")
    with col3:
        st.write(f"PyG Graph: {'✅' if pyg_exists else '❌'}")

    if not all([model_exists, pdf_exists, pyg_exists]):
        missing = []
        if not model_exists:
            missing.append(f"Model file at {MODEL_PATH}")
        if not pdf_exists:
            missing.append(f"PDF file at {PDF_PATH}")
        if not pyg_exists:
            missing.append(f"PyG graph file at {PYG_FILE}")
        st.error(f"Missing required files: {', '.join(missing)}")
        return

    # Analysis Pipeline
    st.header("Analysis Pipeline")

    # Step 1: Load model and get species
    with st.expander("Step 1: Load Model and Get Species", expanded=True):
        if st.button("Load Model", key="load_model"):
            with st.spinner("Loading model and extracting species..."):
                species_list = load_model_and_get_species(MODEL_PATH)
                if species_list:
                    st.session_state["species_list"] = species_list
                    st.success(f"Found {len(species_list)} species in the model")
                    st.write("Species:", species_list)

                    # Extract compartment info
                    compartments = extract_compartment_info(species_list)
                    if compartments:
                        st.success(
                            f"Extracted {len(compartments)} compartments: {', '.join(compartments)}"
                        )
                        # Display compartment data
                        df_compartments = pd.DataFrame(COMPARTMENTS_DATA)
                        st.write("Compartment Information:")
                        st.dataframe(df_compartments)

                    # Extract protein names for UniProt
                    proteins = extract_protein_names(species_list)
                    if proteins:
                        st.success(
                            f"Extracted {len(proteins)} protein names: {', '.join(proteins)}"
                        )
                        st.session_state["proteins"] = proteins
                else:
                    st.error("Failed to extract species from the model")

    # Step 2: Generate descriptions for species
    with st.expander("Step 2: Generate Species Descriptions", expanded=True):
        if (
            st.button("Generate Descriptions", key="gen_desc")
            and "species_list" in st.session_state
        ):
            with st.spinner("Initializing OpenAI components..."):
                text_embedding_model = OpenAIEmbeddings()

            # Generate descriptions for each species
            all_species_descriptions = {}
            progress_bar = st.progress(0)
            species_container = st.container()

            for i, species in enumerate(st.session_state["species_list"]):
                species_container.write(
                    f"Processing species {i+1}/{len(st.session_state['species_list'])}: {species}"
                )
                try:
                    with st.spinner(f"Generating description for {species}..."):
                        description = query_species_description(
                            PDF_PATH, text_embedding_model, species
                        )
                        all_species_descriptions[species] = description
                        species_container.write(f"✅ {species}: Description generated")
                except Exception as e:
                    species_container.write(f"❌ Error processing {species}: {e}")
                    all_species_descriptions[species] = f"Error: {e}"

                # Update progress bar
                progress_bar.progress((i + 1) / len(st.session_state["species_list"]))

            # Clean species descriptions
            with st.spinner("Cleaning species descriptions..."):
                cleaned_species_descriptions = {
                    species: description.replace("\n", " ").strip()
                    for species, description in all_species_descriptions.items()
                }

                # Save in session state
                st.session_state["species_descriptions"] = cleaned_species_descriptions

                # Display species descriptions
                st.subheader("Species Descriptions")

                # FIX: Instead of nesting expanders, use a selectbox to choose the species
                species_names = list(cleaned_species_descriptions.keys())
                selected_species = st.selectbox(
                    "Select a species to view description:", species_names
                )
                if selected_species:
                    st.write(cleaned_species_descriptions[selected_species])

                # Save to JSON
                if st.button("Save Descriptions to JSON", key="save_desc"):
                    output_file = current_dir / "species_descriptions.json"
                    with open(output_file, "w") as f:
                        json.dump(cleaned_species_descriptions, f, indent=4)
                    st.success(f"Saved species descriptions to {output_file}")

    # Step 3: Generate embeddings and map to knowledge graph
    with st.expander(
        "Step 3: Generate Embeddings and Map to Knowledge Graph", expanded=True
    ):
        if (
            st.button("Generate Embeddings and Find Matches", key="gen_embed")
            and "species_descriptions" in st.session_state
        ):
            # Generate embeddings
            with st.spinner("Generating embeddings for species descriptions..."):
                embeddings = generate_embeddings(
                    st.session_state["species_descriptions"]
                )
                st.session_state["embeddings"] = embeddings
                st.success("Embeddings generated")

            # Load PyG data
            with st.spinner("Loading knowledge graph..."):
                try:
                    with open(PYG_FILE, "rb") as f:
                        pyg_data = pickle.load(f)
                    st.success("Knowledge graph loaded")

                    # Display PyG data info
                    st.subheader("Knowledge Graph Data Information")
                    if hasattr(pyg_data, "num_nodes"):
                        st.write(f"Number of nodes: {pyg_data.num_nodes}")
                    if hasattr(pyg_data, "num_edges"):
                        st.write(f"Number of edges: {pyg_data.num_edges}")

                    # Try two approaches to find similar nodes
                    st.subheader("Node Mapping Results")
                    tab1, tab2 = st.tabs(
                        [
                            "Approach 1: Aggregate Similarity",
                            "Approach 2: Individual Matching",
                        ]
                    )

                    with tab1:
                        # Approach 1: Find similar nodes using aggregate similarity
                        node_ids = find_similar_nodes(embeddings, pyg_data)
                        if node_ids:
                            st.success(f"Found {len(node_ids)} matching node IDs")
                            st.write("Node IDs:", node_ids)
                            st.session_state["node_ids"] = node_ids

                    with tab2:
                        # Approach 2: Find similar nodes for each species individually
                        node_id_mapping = get_species_node_mapping(
                            st.session_state["species_list"], embeddings, pyg_data
                        )
                        if node_id_mapping:
                            # Convert to DataFrame for better display
                            df_mapping = pd.DataFrame(
                                list(node_id_mapping.items()),
                                columns=["Species", "Node ID"],
                            )
                            st.success(
                                f"Found {len(node_id_mapping)} species-to-node mappings"
                            )
                            st.dataframe(df_mapping)
                            st.session_state["node_id_mapping"] = node_id_mapping

                except Exception as e:
                    st.error(f"Error loading or processing knowledge graph: {e}")
                    st.code(traceback.format_exc())

    # Step 4: Query PrimeKG and UniProt
    with st.expander("Step 4: Query PrimeKG and UniProt", expanded=True):
        if st.button("Query PrimeKG", key="query_primekg") and (
            "node_ids" in st.session_state or "node_id_mapping" in st.session_state
        ):
            with st.spinner("Querying PrimeKG..."):
                try:
                    # Initialize PrimeKG
                    primekg_data = PrimeKG(local_dir=PRIMEKG_DIR)
                    primekg_data.load_data()

                    # Get all nodes
                    primekg_nodes = primekg_data.get_nodes()

                    # Approach 1: Use aggregate node IDs
                    if "node_ids" in st.session_state and st.session_state["node_ids"]:
                        node_ids = st.session_state["node_ids"]
                        final_nodes_1 = primekg_nodes[
                            primekg_nodes["node_index"].isin(node_ids)
                        ]
                        st.subheader("PrimeKG Results (Approach 1)")
                        st.success(
                            f"Found {len(final_nodes_1)} matching nodes in PrimeKG"
                        )
                        if not final_nodes_1.empty:
                            st.dataframe(final_nodes_1)

                            # Save to CSV
                            if st.button(
                                "Save Results to CSV (Approach 1)", key="save_csv1"
                            ):
                                output_file = (
                                    current_dir / "matching_nodes_approach1.csv"
                                )
                                final_nodes_1.to_csv(output_file, index=False)
                                st.success(f"Saved matching nodes to {output_file}")

                    # Approach 2: Use individual node ID mapping
                    if (
                        "node_id_mapping" in st.session_state
                        and st.session_state["node_id_mapping"]
                    ):
                        node_id_mapping = st.session_state["node_id_mapping"]
                        # Create a dataframe with species and node IDs
                        df_node_id = pd.DataFrame(
                            list(node_id_mapping.items()),
                            columns=["specie_name", "node_index"],
                        )

                        # Merge with primekg_nodes to get all node info
                        merged_df = pd.merge(
                            df_node_id, primekg_nodes, on="node_index", how="inner"
                        )

                        st.subheader("PrimeKG Results (Approach 2)")
                        st.success(
                            f"Found {len(merged_df)} matching nodes in PrimeKG with species mapping"
                        )
                        if not merged_df.empty:
                            st.dataframe(merged_df)

                            # Save to CSV
                            if st.button(
                                "Save Results to CSV (Approach 2)", key="save_csv2"
                            ):
                                output_file = (
                                    current_dir / "matching_nodes_approach2.csv"
                                )
                                merged_df.to_csv(output_file, index=False)
                                st.success(f"Saved matching nodes to {output_file}")

                except Exception as e:
                    st.error(f"Error querying PrimeKG: {e}")
                    st.code(traceback.format_exc())

        if (
            st.button("Query UniProt", key="query_uniprot")
            and "proteins" in st.session_state
        ):
            with st.spinner("Querying UniProt..."):
                try:
                    # Get a list of gene IDs (hardcoded for now, you'll need to extract these from your data)
                    # In a real app, you would extract these from the node data or another source
                    gene_ids = ["6774", "3569", "3586", "1401"]  # Example gene IDs

                    # Submit ID mapping job
                    job_id = submit_id_mapping(
                        from_db="GeneID", to_db="UniProtKB", ids=gene_ids
                    )
                    st.write(f"Submitted UniProt ID mapping job: {job_id}")

                    # Check job status
                    with st.spinner("Waiting for UniProt results..."):
                        if check_id_mapping_results_ready(job_id):
                            # Get results
                            link = get_id_mapping_results_link(job_id)
                            mapping_results = get_id_mapping_results_stream(link)

                            if mapping_results and "results" in mapping_results:
                                # Convert mapping results to a dataframe
                                protein_mapped_df = pd.DataFrame(
                                    mapping_results["results"]
                                )
                                st.success("Retrieved UniProt mapping results")

                                # Process results to get Swiss-Prot entries
                                if not protein_mapped_df.empty:
                                    # Filter for reviewed entries (Swiss-Prot)
                                    protein_reviewed_df = protein_mapped_df[
                                        protein_mapped_df.apply(
                                            lambda x: x["to"]["entryType"]
                                            == "UniProtKB reviewed (Swiss-Prot)",
                                            axis=1,
                                        )
                                    ]
                                    protein_reviewed_df.reset_index(
                                        drop=True, inplace=True
                                    )

                                    # Extract fields from the 'to' column
                                    for key in protein_reviewed_df["to"][0].keys():
                                        protein_reviewed_df[key] = [
                                            x[key] if key in x else "N/A"
                                            for x in protein_reviewed_df["to"]
                                        ]

                                    st.subheader("UniProt Results")
                                    st.dataframe(protein_reviewed_df)

                                    # Save to CSV
                                    if st.button(
                                        "Save UniProt Results to CSV",
                                        key="save_uniprot",
                                    ):
                                        output_file = (
                                            current_dir / "uniprot_results.csv"
                                        )
                                        protein_reviewed_df.to_csv(
                                            output_file, index=False
                                        )
                                        st.success(
                                            f"Saved UniProt results to {output_file}"
                                        )

                except Exception as e:
                    st.error(f"Error querying UniProt: {e}")
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
