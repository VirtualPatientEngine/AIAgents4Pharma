import fitz  # PyMuPDF
import openai
import json
from tqdm import tqdm
import pandas as pd
import sys
import os
from dotenv import load_dotenv

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to query the GPT API
def query_gpt(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        res = response['choices'][0]['message']['content'].strip()
        res = json.loads(res)
        return res['description']
    except Exception as e:
        return 'No description found'

# Main function
def main(pdf_path,names,output_file='descriptions_output.json'):
    # Extract text from the PDF
    paper_text = extract_text_from_pdf(pdf_path)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    descs = {}
    # Loop over each name and query the GPT API
    for name in tqdm(names, desc='Extracting descriptions'):
        prompt = (
            f"JUST ANSWER WITH THE DESCRIPTION WE ARE LOOKING FOR. "
            f"Extract the description of {name} species from this paper:\n\n"
            f"{paper_text}\n\n"
            f"DO NOT PUT ANY MORE INFO THAN THE SPECIES DESCRIPTION. "
            f"Let the response be JSON with key 'description'."
        )
        description = query_gpt(prompt, api_key)
        descs[name] = description
        print(f"Species: {name}")
        print(f"Description: {description}")
    #write to file
    with open(output_file, 'w') as f:
        json.dump(descs, f)
    

if __name__ == '__main__':
    args = sys.argv[1:]
    names_path = 'model537_mapping.xlsx'
    names_df = pd.read_excel(names_path)
    names_df.columns = names_df.columns.str.strip()
    names_df = names_df[names_df['Species Name'].notnull()]
    names = names_df['Species Name'].tolist()
    main(args[0],names)
