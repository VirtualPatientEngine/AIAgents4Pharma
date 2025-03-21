import requests
import json

def fetch_paper_metadata(semantic_scholar_id, api_key=None):
    # API endpoint for Semantic Scholar Graph API
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{semantic_scholar_id}"
    
    # Define the fields we want to retrieve
    params = {
        "fields": "title,authors,abstract,year,venue,publicationDate,citationCount,referenceCount,openAccessPdf"
    }
    
    # Headers for API key (optional)
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        # Make the API request
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        paper_data = response.json()
        
        # Extract and display metadata
        metadata = {
            "Title": paper_data.get("title", "N/A"),
            "Authors": [author["name"] for author in paper_data.get("authors", [])] if paper_data.get("authors") else "N/A",
            "Abstract": paper_data.get("abstract", "N/A"),
            "Year": paper_data.get("year", "N/A"),
            "Venue": paper_data.get("venue", "N/A"),
            "Publication Date": paper_data.get("publicationDate", "N/A"),
            "Citation Count": paper_data.get("citationCount", "N/A"),
            "Reference Count": paper_data.get("referenceCount", "N/A"),
            "Open Access PDF": paper_data.get("openAccessPdf", {}).get("url", "Not available") if paper_data.get("openAccessPdf") else "Not available"
        }
        
        # Pretty print the metadata
        print("Paper Metadata:")
        for key, value in metadata.items():
            if key == "Authors":
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        
        return metadata
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            print("Too Many Requests. Please wait and try again or use an API key for higher rate limits.")
            print("API Key Form: https://www.semanticscholar.org/product/api#api-key-form")
        else:
            print(f"HTTP Error occurred: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error occurred: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def main():
    # Paper ID from the URL
    paper_id = "94f5b9d3028797d4ed59cafbb302c8acce1f95b1"
    
    # Optional: Add your API key here if you have one
    api_key = None  # Replace with "YOUR_API_KEY_HERE" if you have a key
    
    print(f"Fetching metadata for paper ID: {paper_id}")
    fetch_paper_metadata(paper_id, api_key)

if __name__ == "__main__":
    main()