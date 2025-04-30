import pandas as pd 

def flatten_species_dict(species_dict_ont):
    rows = []

    # Iterate through the dictionary to extract data
    for key, value in species_dict_ont.items():
        for annotation in value['ontology_annotations']:
            row = {
                "species": key,
                "text": annotation['text'],
                "id": annotation['id'],
                "uri": annotation['uri'],
                "source_ontology": annotation['source_ontology']
            }
            rows.append(row)

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    df['id'] = df['id'].astype(str).str.extract('(\\d+)$')[0].astype(int).astype(str)
    return df

