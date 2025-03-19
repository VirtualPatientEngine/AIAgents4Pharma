import os
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def get_disease_names_from_text(text):
    """Uses GPT-4o Mini API to extract disease names from text."""
    if not OPENAI_API_KEY:
        raise ValueError("Missing OpenAI API key. Ensure it's set in the .env file.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
    The following text is extracted from a scientific paper. Identify the 3 high level Gene Ontologies related the text form below terms:
    e.g: Metabolism, immune system process, plasma membrane, mitochondria, etc.

    
    Extracted Paper Text:
    {text}

    **Return only keywords as a comma-separated list (no explanations).**
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    extracted_diseases = response.choices[0].message.content.split(", ")
    return extracted_diseases if extracted_diseases else ["No known diseases found"]

def extract_diseases_from_pdf(pdf_path):
    """Main function to extract disease names from a PDF."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return "No text found in PDF."

    disease_names = get_disease_names_from_text(text)
    return disease_names

def main():
    # path to pdf 
    pdf = 'psp201364a.pdf' # Replace with the path to PDF, may be better method for this

    diseases = extract_diseases_from_pdf(pdf)

    print(diseases)

if __name__ == "__main__":
    main()  