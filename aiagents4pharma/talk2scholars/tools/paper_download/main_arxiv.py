import os
from arxiv_downloader import ArxivPaperDownloader

if __name__ == "__main__":
    # Replace with a real arXiv ID:
    arxiv_id = "2303.10163"
    downloader = ArxivPaperDownloader()

    try:
        # Fetch metadata
        metadata = downloader.fetch_metadata(arxiv_id)
        print("Metadata fetched successfully.")
        # Preview the first 500 characters of the metadata
        print(metadata["xml"][:500])

        # Download PDF
        pdf_data = downloader.download_pdf(arxiv_id)
        print(f"PDF downloaded from: {pdf_data['pdf_url']}")

        # Determine the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(script_dir, f"{arxiv_id}.pdf")

        # Save the PDF to a file
        with open(pdf_path, "wb") as f:
            f.write(pdf_data["pdf_object"])
        print(f"PDF saved as {pdf_path}")

    except Exception as e:
        print("Error:", str(e))
