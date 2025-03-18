import requests
import json
import urllib.request
import urllib.parse
import urllib.error
import copy
import re


def get_bioportal_annotations(text, api_key, ontologies=None, longest_only=True, max_level=0, include=None):
    """
    Get annotations for text using BioPortal Annotator API.
    
    Parameters:
    text (str): The text to annotate
    api_key (str): Your BioPortal API key
    ontologies (list, optional): List of ontology acronyms to use for annotation
    longest_only (bool, optional): Whether to return only the longest annotation span
    max_level (int, optional): Maximum level in hierarchy to return (0 for none)
    include (list, optional): Additional fields to include (e.g., prefLabel, synonym, definition)
    
    Returns:
    dict: JSON response from the BioPortal Annotator API
    """
    # Base URL for the BioPortal Annotator
    base_url = "https://data.bioontology.org/annotator"
    
    # Set up parameters
    params = {
        "apikey": api_key,
        "text": text,
        "longest_only": str(longest_only).lower(),
        "format": "json"
    }
    
    # Add ontologies parameter if specified
    if ontologies:
        params["ontologies"] = ",".join(ontologies)
    
    # Add hierarchy level if specified
    if max_level > 0:
        params["max_level"] = str(max_level)
    
    # Add additional fields to include if specified
    if include:
        if isinstance(include, list):
            params["include"] = ",".join(include)
        else:
            params["include"] = include
    
    # Make the request
    response = requests.get(base_url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def get_json_with_auth(url, api_key):
    """
    Get JSON from URL with authentication header.
    Alternative method using urllib as shown in BioOntology example.
    
    Parameters:
    url (str): URL to request
    api_key (str): Your BioPortal API key
    
    Returns:
    dict: JSON response
    """
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + api_key)]
    try:
        return json.loads(opener.open(url).read())
    except urllib.error.HTTPError as e:
        print(f"Error: {e.code}")
        print(e.read())
        return None


def display_annotations(annotations, api_key=None, fetch_class_details=False):
    """
    Display annotations in a more readable format.
    
    Parameters:
    annotations (list): List of annotations from BioPortal
    api_key (str, optional): API key to fetch additional class details
    fetch_class_details (bool, optional): Whether to fetch detailed class information
    """
    if not annotations:
        print("No annotations found.")
        return
    
    print(f"Found {len(annotations)} annotations:")
    print("-" * 60)
    
    for i, result in enumerate(annotations, 1):
        print(f"Annotation #{i}:")
        
        # Get class details
        class_details = result["annotatedClass"]
        if fetch_class_details and api_key:
            try:
                class_details = get_json_with_auth(class_details["links"]["self"], api_key)
            except (KeyError, urllib.error.HTTPError) as e:
                print(f"  Error retrieving class details: {e}")
        
        # Print class details
        print("  Class details:")
        print(f"    ID: {class_details.get('@id', 'N/A')}")
        print(f"    Preferred Label: {class_details.get('prefLabel', 'N/A')}")
        
        # Print ontology info if available
        if "links" in class_details and "ontology" in class_details["links"]:
            ontology_link = class_details["links"]["ontology"]
            ontology_id = ontology_link.split("/")[-1] if "/" in ontology_link else ontology_link
            print(f"    Ontology: {ontology_id}")
        
        # Print additional details if available
        if "synonym" in class_details:
            print(f"    Synonyms: {', '.join(class_details['synonym'])}")
        if "definition" in class_details:
            print(f"    Definition: {class_details['definition'][0] if class_details['definition'] else 'N/A'}")
        
        # Print annotation details
        print("  Annotation details:")
        for annotation in result.get("annotations", []):
            print(f"    From: {annotation.get('from', 'N/A')}")
            print(f"    To: {annotation.get('to', 'N/A')}")
            print(f"    Match type: {annotation.get('matchType', 'N/A')}")
            print(f"    Text: {annotation.get('text', 'N/A')}")
        
        # Print hierarchy details if available
        if result.get("hierarchy"):
            print("\n  Hierarchy annotations:")
            for hierarchy_item in result["hierarchy"]:
                hier_class = hierarchy_item["annotatedClass"]
                distance = hierarchy_item.get("distance", "N/A")
                
                if fetch_class_details and api_key:
                    try:
                        hier_class = get_json_with_auth(hier_class["links"]["self"], api_key)
                    except (KeyError, urllib.error.HTTPError) as e:
                        print(f"    Error retrieving hierarchy class: {e}")
                        continue
                
                print(f"    Parent/Child class (distance {distance}):")
                print(f"      ID: {hier_class.get('@id', 'N/A')}")
                print(f"      Preferred Label: {hier_class.get('prefLabel', 'N/A')}")
        
        print("-" * 60)


def extract_id_from_uri(uri):
    """
    Extract a short ID from a URI, typically the part after the last slash or after the last colon.
    
    Parameters:
    uri (str): The URI string
    
    Returns:
    str: The extracted ID
    """
    # Try different patterns to extract the ID
    if not uri:
        return "N/A"
    
    # Handle URLs with format http://purl.obolibrary.org/obo/GO_0016020
    obo_match = re.search(r'/obo/([^/]+)$', uri)
    if obo_match:
        return obo_match.group(1)
    
    # Handle other URL formats - get the last component
    uri_parts = uri.rstrip('/').split('/')
    if uri_parts:
        return uri_parts[-1]
    
    return uri


def enrich_species_dict_with_ontologies(species_dict, api_key, ontologies=None):
    """
    Enrich the species dictionary with ontology annotations from BioPortal.
    
    Parameters:
    species_dict (dict): Dictionary of species with 'name' and 'description' fields
    api_key (str): Your BioPortal API key
    ontologies (list, optional): List of ontology acronyms to use for annotation
    
    Returns:
    dict: Enriched dictionary with ontology annotations
    """
    # Create a deep copy of the original dictionary to avoid modifying it
    enriched_dict = copy.deepcopy(species_dict)
    
    # Extract all names from the dictionary
    names_list = []
    name_to_key_map = {}
    
    for key, entity in species_dict.items():
        name = entity.get('name', '')
        if name:
            names_list.append(name)
            name_to_key_map[name] = key
    
    # Join all the names with newlines to create a single text
    if not names_list:
        print("No names found in the dictionary to annotate.")
        return enriched_dict
    
    combined_text = '\n'.join(names_list)
    print(f"Combined {len(names_list)} names for annotation.")
    
    # Get annotations from BioPortal in a single call
    annotations = get_bioportal_annotations(
        text=combined_text,
        api_key=api_key,
        ontologies=ontologies,
        longest_only=True,
        include=["prefLabel"]
    )
    
    # Initialize ontology_annotations for all entities
    for key in enriched_dict:
        enriched_dict[key]['ontology_annotations'] = []
    
    # Process the annotations
    if not annotations:
        print("No annotations found for any names.")
        return enriched_dict
    
    print(f"Received {len(annotations)} annotations from BioPortal.")
    
    # Map annotations to the right dictionary entries
    for annotation in annotations:
        # Get the relevant annotation details
        for ann_detail in annotation.get('annotations', []):
            from_pos = ann_detail.get('from', 0)
            to_pos = ann_detail.get('to', 0)
            matched_text = ann_detail.get('text', '')
            
            # Find which name this annotation belongs to
            target_name = None
            curr_pos = 0
            
            for name in names_list:
                name_end = curr_pos + len(name)
                # Check if the annotation is within this name's text range
                if from_pos >= curr_pos and to_pos <= name_end:
                    target_name = name
                    break
                # Move to next name position (add 1 for the newline)
                curr_pos = name_end + 1
            
            if target_name:
                # Get the corresponding dictionary key
                dict_key = name_to_key_map.get(target_name)
                
                if dict_key:
                    # Get class details from the annotation
                    class_details = annotation.get('annotatedClass', {})
                    uri = class_details.get('@id', '')
                    
                    # Get the ontology ID
                    ontology_id = None
                    if 'links' in class_details and 'ontology' in class_details['links']:
                        ontology_link = class_details['links']['ontology']
                        ontology_id = ontology_link.split("/")[-1] if "/" in ontology_link else ontology_link
                    
                    # Add to the enriched dictionary
                    enriched_dict[dict_key]['ontology_annotations'].append({
                        'text': matched_text,
                        'id': extract_id_from_uri(uri),
                        'uri': uri,
                        'source_ontology': ontology_id
                    })
    
    # Report statistics
    annotation_counts = {k: len(v.get('ontology_annotations', [])) for k, v in enriched_dict.items()}
    total_annotations = sum(annotation_counts.values())
    
    print(f"Added a total of {total_annotations} annotations to {len([c for c in annotation_counts.values() if c > 0])} entries.")
    
    return enriched_dict
