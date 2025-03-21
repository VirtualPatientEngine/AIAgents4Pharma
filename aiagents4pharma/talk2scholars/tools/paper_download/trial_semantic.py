import requests
import os
import time

def download_paper(semantic_scholar_id, max_retries=3):
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{semantic_scholar_id}"
    params = {"fields": "title,openAccessPdf"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            paper_data = response.json()
            
            if 'openAccessPdf' in paper_data and paper_data['openAccessPdf'] is not None:
                pdf_url = paper_data['openAccessPdf']['url']
                title = paper_data.get('title', 'paper').replace(' ', '_').replace('/', '_')
                filename = f"{title}_{semantic_scholar_id}.pdf"
                
                print(f"Found open access PDF. Downloading: {title}")
                pdf_response = requests.get(pdf_url)
                pdf_response.raise_for_status()
                
                with open(filename, 'wb') as f:
                    f.write(pdf_response.content)
                
                print(f"Paper successfully downloaded as: {filename}")
                return True
            else:
                print("This paper does not have an open access PDF available.")
                return False
                
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                wait_time = 60  # Wait 1 minute
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Error occurred: {str(e)}")
                return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return False
    
    print("Max retries reached. Please try again later or use an API key.")
    return False

def main():
    paper_id = "55247d4fd2e62ec6bce98c7d5563dd4cebc315c5"
    print(f"Attempting to download paper with ID: {paper_id}")
    download_paper(paper_id)

if __name__ == "__main__":
    main()