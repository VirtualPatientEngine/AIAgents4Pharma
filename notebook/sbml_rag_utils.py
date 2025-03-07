import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import libsbml
import re


def analyze_model_species(sbml_file_path):
    """
    Analyzes the SBML model and returns statistics about species in the model.
    
    Args:
        sbml_file_path (str): Path to the SBML model file
        
    Returns:
        dict: Statistics about species in the model
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbml_file_path)
    
    # Check for errors
    if document.getNumErrors() > 0:
        print(f"Errors in SBML document: {document.getNumErrors()}")
        return {"error": f"Errors in SBML document: {document.getNumErrors()}"}
    
    model = document.getModel()
    
    # Get species count
    species_count = model.getNumSpecies()
    
    # Get species details
    species_details = []
    compartment_distribution = {}
    duplicate_names = {}
    
    for s in range(species_count):
        species_obj = model.getSpecies(s)
        species_id = species_obj.getId()
        species_name = species_obj.getName() or species_id
        compartment = species_obj.getCompartment()
        
        # Track duplicated names
        if species_name in duplicate_names:
            duplicate_names[species_name] += 1
        else:
            duplicate_names[species_name] = 1
            
        # Track compartment distribution
        if compartment in compartment_distribution:
            compartment_distribution[compartment] += 1
        else:
            compartment_distribution[compartment] = 1
            
        # Track if species has initial concentration or amount
        has_initial_amount = species_obj.isSetInitialAmount()
        has_initial_concentration = species_obj.isSetInitialConcentration()
        
        species_details.append({
            'id': species_id,
            'name': species_name,
            'compartment': compartment,
            'has_initial_amount': has_initial_amount,
            'has_initial_concentration': has_initial_concentration
        })
    
    # Filter duplicate names
    duplicate_name_counts = {name: count for name, count in duplicate_names.items() if count > 1}
    
    # Get model units if available
    substance_units = model.getSubstanceUnits() if model.isSetSubstanceUnits() else "Not specified"
    time_units = model.getTimeUnits() if model.isSetTimeUnits() else "Not specified"
    
    # Generate summary statistics
    statistics = {
        "total_species_count": species_count,
        "compartment_distribution": compartment_distribution,
        "substance_units": substance_units,
        "time_units": time_units,
        "species_details": species_details,
        "duplicate_names": duplicate_name_counts,
        "duplicate_name_count": len(duplicate_name_counts),
        "species_with_initial_amount": sum(1 for s in species_details if s['has_initial_amount']),
        "species_with_initial_concentration": sum(1 for s in species_details if s['has_initial_concentration'])
    }
    
    return statistics

def load_and_process_pdf(pdf_path):
    """Load PDF and create searchable vector database"""
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore, documents

def extract_keywords_from_pdf(documents, num_keywords=20):
    """Extract the top N biomedical entity keywords from the PDF
    
    Args:
        documents: List of documents to extract keywords from
        num_keywords: Number of keywords to extract (default: 10)
        
    Returns:
        List of keywords with standardized format (uses abbreviations, removes parentheses)
    """
    llm = ChatOpenAI(model="gpt-4o")
    
    # Combine all document content
    full_text = ""
    for doc in documents:
        full_text += doc.page_content + "\n"
    
    # Create a prompt for keyword extraction focusing on biomedical entities
    prompt = PromptTemplate.from_template(
        """You are given a scientific document related to a systems biology model.
        Extract exactly {num_keywords} key biomedical entities that best represent the main focus of this document.
        
        FOCUS ONLY ON:
        - Disease names (e.g., Alzheimer's disease, cancer, diabetes)
        - Gene or protein names (e.g., TNF-alpha, P53, BRCA1)
        - Pathways or biological processes (e.g., apoptosis, inflammation)
        - Cell types or tissues (e.g., T-cells, hepatocytes)
        - Drug classes or specific drugs (e.g., statins, metformin)
        
        DO NOT INCLUDE general methodologies, modeling approaches, or broad fields of study.
        
        IMPORTANT FORMATTING RULES:
        1. When an entity has both a full name and abbreviation, use ONLY the abbreviation (e.g., use "IL-6" instead of "Interleukin-6")
        2. Do not include any parentheses in your output
        3. Do not include the full name with the abbreviation (e.g., do not return "Interleukin-6 (IL-6)")
        
        Format your response as a comma-separated list.
        Document:
        {text}
        
        {num_keywords} BIOMEDICAL ENTITY KEYWORDS:"""
    )
    
    # Create and run the chain
    chain = prompt | llm | StrOutputParser()
    keywords = chain.invoke({"text": full_text[:15000], "num_keywords": num_keywords})  # Increased char limit to get more context
    
    # Process the result to get a clean list
    keyword_list = [k.strip() for k in keywords.split(',')]
    
    # Additional processing to standardize format
    cleaned_keywords = []
    for keyword in keyword_list:
        # Remove any remaining parenthetical content
        cleaned = re.sub(r'\s*\([^)]*\)', '', keyword)
        # Remove any special characters except hyphens and spaces
        cleaned = re.sub(r'[^\w\s\-]', '', cleaned)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:  # Only add non-empty strings
            cleaned_keywords.append(cleaned)
    
    # Ensure we get exactly the requested number of keywords (if available)
    return cleaned_keywords[:num_keywords]

def parse_sbml_model(sbml_file_path):
    """Parse SBML model and extract species only, handling duplicate names"""
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbml_file_path)
    
    # Check for errors
    if document.getNumErrors() > 0:
        print(f"Errors in SBML document: {document.getNumErrors()}")
        return []
    
    model = document.getModel()
    
    # Extract species (molecules, proteins, etc.)
    species = []
    species_names = {}  # Track name occurrence count
    
    for s in range(model.getNumSpecies()):
        species_obj = model.getSpecies(s)
        species_id = species_obj.getId()
        species_name = species_obj.getName() or species_id
        compartment = species_obj.getCompartment()
        
        # Check if this name already exists and append compartment or count if needed
        if species_name in species_names:
            species_names[species_name] += 1
            # Make the name unique by adding compartment and count
            unique_name = f"{species_name} ({compartment}, #{species_names[species_name]})"
        else:
            species_names[species_name] = 1
            unique_name = species_name
        
        species.append({
            'id': species_id,
            'name': unique_name,
            'original_name': species_name,
            'compartment': compartment,
            'type': 'species'
        })
    
    return species

def get_species_background(species, vectorstore):
    """Get background information for a specific species"""
    llm = ChatOpenAI(model="gpt-4o")
    
    # Create a retriever for relevant information
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Use original name for searching but include ID for clarity
    search_name = species.get('original_name', species['name'])
    display_name = species['name']
    species_id = species['id']
    compartment = species.get('compartment', 'unknown compartment')
    
    # Define the RAG prompt
    template = """You are a systems biology expert assistant. Based on the provided context, 
    extract and summarize relevant background information about the species named '{species_name}' 
    (ID: {species_id}, located in {compartment}).
    
    Focus on information about this specific biological species/molecule, including:
    - Its biological function
    - Its role in pathways
    - Any interactions with other molecules
    - Its importance in the biological system being modeled
    
    If there is no specific information about this species in the context, infer what it might be based on 
    general knowledge about similar biological components in the context described.
    
    Context:
    {context}
    
    Background information about {species_name}:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # The query to search for in the vectorstore - use original name for better matching
    query_text = f"Information about {search_name}"
    
    # Create the RAG pipeline with correct input handling
    rag_chain = (
        {
            "context": retriever, 
            "species_name": lambda _: display_name, 
            "species_id": lambda _: species_id,
            "compartment": lambda _: compartment
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run the chain with the correct string query instead of a dict
    background = rag_chain.invoke(query_text)
    
    return {
        "id": species_id,
        "name": display_name,
        "original_name": search_name,
        "compartment": compartment,
        "background": background
    }

def process_sbml_and_pdf(sbml_file_path, pdf_file_path, api_key=None, max_species=None):
    """Main function to process SBML model and extract background information for species"""
    # Set OpenAI API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # First, analyze the model to get species statistics
    print("Analyzing SBML model species...")
    species_stats = analyze_model_species(sbml_file_path)
    print(f"Found {species_stats['total_species_count']} species in the model")
    print(f"Species are distributed across {len(species_stats['compartment_distribution'])} compartments")
    
    # Load and process PDF
    print("Loading and processing PDF...")
    vectorstore, documents = load_and_process_pdf(pdf_file_path)
    
    # Extract keywords from PDF
    print("Extracting keywords from PDF...")
    keywords = extract_keywords_from_pdf(documents)
    
    # Parse SBML model to get species
    print("Parsing SBML model for species...")
    species_list = parse_sbml_model(sbml_file_path)
    
    # Limit the number of species to process if specified
    if max_species and max_species < len(species_list):
        print(f"Limiting processing to {max_species} species (out of {len(species_list)} total)")
        species_list = species_list[:max_species]
    
    # Get background information for each species
    print(f"Extracting background information for {len(species_list)} species...")
    species_backgrounds = []
    for i, species in enumerate(species_list):
        print(f"Processing species {i+1}/{len(species_list)}: {species['name']}")
        background_info = get_species_background(species, vectorstore)
        species_backgrounds.append(background_info)
    
    return species_backgrounds, keywords, species_stats