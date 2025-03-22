import os
from pubmed_downloader import PubMedPaperDownloader

if __name__ == "__main__":
    pmid = "26572668 "  # Replace with an actual PubMed ID (PMID)
    downloader = PubMedPaperDownloader()

    try:
        metadata = downloader.fetch_metadata(pmid)
        print("Metadata fetched successfully.")
        print(metadata["xml"][:500])  # Preview first 500 characters

        pdf_data = downloader.download_pdf(pmid)
        print(f"PDF downloaded from: {pdf_data['pdf_url']}")

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(script_dir, f"{pmid}.pdf")

        # Save to file
        with open(pdf_path, "wb") as f:
            f.write(pdf_data["pdf_object"])
        print(f"PDF saved as {pdf_path}")

    except Exception as e:
        print("Error:", str(e))
