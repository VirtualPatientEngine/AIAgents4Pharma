import spacy
import scispacy
from scispacy.linking import EntityLinker

def extract_umls_codes(text):
    """
    Extract UMLS codes from medical text using ScispaCy.
    
    Parameters:
    -----------
    text : str
        The medical text to extract UMLS codes from
    
    Returns:
    --------
    list
        A list of dictionaries containing extracted entities, their UMLS CUIs, and other metadata
    """
    # Load the ScispaCy model with UMLS linking
    # You can choose from different models: "en_core_sci_sm", "en_core_sci_md", or "en_core_sci_lg"
    # Larger models are more accurate but require more memory and processing time
    try:
        nlp = spacy.load("en_core_sci_md")
    except OSError:
        print("Model not found. Installing the model...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz"])
        nlp = spacy.load("en_core_sci_md")
    
    # Add the UMLS entity linker to the pipeline
    if "scispacy_linker" not in nlp.pipe_names:
        # Initialize the entity linker with UMLS vocabulary
        linker = EntityLinker(resolve_abbreviations=True, 
                              name="umls", 
                              threshold=0.7)  # Adjust threshold as needed
        nlp.add_pipe("scispacy_linker", config={"linker": linker})
    
    # Process the text
    doc = nlp(text)
    
    # Extract entities and their UMLS codes
    results = []
    for entity in doc.ents:
        # Get UMLS links for the entity
        umls_links = []
        if entity._.kb_ents:
            # Extract top UMLS matches
            for umls_ent_id, score in entity._.kb_ents:
                cui = entity._.kb_ents[0][0]
                umls_concept = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui]
                
                umls_links.append({
                    'cui': cui,
                    'score': score,
                    'name': umls_concept['canonical_name'],
                    'types': umls_concept['types']
                })
        
        entity_data = {
            'text': entity.text,
            'label': entity.label_,
            'start': entity.start_char,
            'end': entity.end_char,
            'umls_links': umls_links
        }
        results.append(entity_data)
    
    return results

def print_umls_results(results):
    """
    Pretty print the UMLS extraction results.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing entity and UMLS information
    """
    print(f"Found {len(results)} medical entities:")
    print("-" * 80)
    
    for idx, entity in enumerate(results, 1):
        print(f"{idx}. \"{entity['text']}\" ({entity['label']})")
        
        if entity['umls_links']:
            print(f"   UMLS Codes:")
            for link in entity['umls_links']:
                print(f"   - CUI: {link['cui']}")
                print(f"     Name: {link['name']}")
                print(f"     Score: {link['score']:.4f}")
                if link['types']:
                    print(f"     Semantic Types: {', '.join(link['types'])}")
                print("")
        else:
            print("   No UMLS codes found for this entity.")
        print("-" * 80)


# Example usage
if __name__ == "__main__":
    # Example medical text
    medical_text = """
    The patient was diagnosed with hypertension and type 2 diabetes mellitus. 
    She was prescribed metformin 500mg twice daily and lisinopril 10mg once daily.
    The patient also has a history of myocardial infarction and chronic kidney disease.
    """
    
    # Extract UMLS codes
    results = extract_umls_codes(medical_text)
    
    # Print results
    print_umls_results(results)


# Batch processing for multiple texts
def process_multiple_texts(texts):
    """
    Process multiple texts and extract UMLS codes from each.
    
    Parameters:
    -----------
    texts : list
        A list of strings containing medical texts
    
    Returns:
    --------
    dict
        A dictionary with text indices as keys and extraction results as values
    """
    results = {}
    
    # Load model once for efficiency
    try:
        nlp = spacy.load("en_core_sci_md")
    except OSError:
        print("Model not found. Installing the model...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz"])
        nlp = spacy.load("en_core_sci_md")
    
    # Add the UMLS entity linker to the pipeline
    if "scispacy_linker" not in nlp.pipe_names:
        linker = EntityLinker(resolve_abbreviations=True, name="umls", threshold=0.7)
        nlp.add_pipe("scispacy_linker", config={"linker": linker})
    
    # Process each text
    for idx, text in enumerate(texts):
        doc = nlp(text)
        text_results = []
        
        for entity in doc.ents:
            umls_links = []
            if entity._.kb_ents:
                for umls_ent_id, score in entity._.kb_ents:
                    cui = entity._.kb_ents[0][0]
                    umls_concept = nlp.get_pipe("scispacy_linker").kb.cui_to_entity[cui]
                    
                    umls_links.append({
                        'cui': cui,
                        'score': score,
                        'name': umls_concept['canonical_name'],
                        'types': umls_concept['types']
                    })
            
            entity_data = {
                'text': entity.text,
                'label': entity.label_,
                'start': entity.start_char,
                'end': entity.end_char,
                'umls_links': umls_links
            }
            text_results.append(entity_data)
        
        results[idx] = text_results
    
    return results