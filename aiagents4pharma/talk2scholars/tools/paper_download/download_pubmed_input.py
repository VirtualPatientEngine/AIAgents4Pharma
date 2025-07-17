#!/usr/bin/env python3
"""
Functionality for downloading PubMedX paper metadata and retrieving the PDF URL.
"""

import xml.etree.ElementTree as ET
import logging
from typing import Any, List
import requests
import hydra

from .base_retreiver import BasePaperRetriever

logger = logging.getLogger(__name__)

class DownloadPubmedPaperInput(BasePaperRetriever):
    """Schema to retrieve paper metadata from PubMed Central"""
    def __init__(self):
        self.metadata_url= None
        self.pdf_base_url= None
        self.map_url= None

    def load_hydra_configs(self):
        """Load Hydra Configurations"""
        logger.info("Loading Hydra Configs for PubMedRetriever")
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="config", overrides=["tools/download_pubmed_paper=default"]
            )
            self.metadata_url = cfg.tools.download_pubmed_paper.metadata_url
            self.pdf_base_url = cfg.tools.download_pubmed_paper.pdf_base_url
            self.map_url = cfg.tools.download_pubmed_paper.map_url

    def fetch_metadata(
        self,
        url: str,
        paper_id: str
    ) -> dict:
        """Fetch and parse metadata from the API url"""
        logger.info("Fetching metadata for paper id %s",paper_id)
        response = requests.get(
        url,
        params={"db": "pmc", "id": paper_id, "retmode": "xml"},
        timeout=10,
        )
        response.raise_for_status()
        return {"data": ET.fromstring(response.text)}

    def extract_metadata(
        self,
        data: dict,
        paper_id: str
    ) -> dict:
        """Extract metadata from the XML entry."""
        logger.info("Extracting metadata for paper id %s",paper_id)
        xml_root = data["data"]
        title_elem = xml_root.find(".//article-title")
        title = title_elem.text if title_elem is not None else ""

        abstract_elem = xml_root.find(".//abstract")
        abstract = "".join(abstract_elem.itertext()).strip() if abstract_elem is not None else ""

        authors = ", ".join(
f"{name.findtext('given-names', default='')} {name.findtext('surname', default='')}".strip()
            for contrib in xml_root.findall('.//contrib[@contrib-type="author"]')
            if (name := contrib.find("name")) is not None
        ) or ""

        pub_date_elem = xml_root.find(".//published")
        pub_date = pub_date_elem.text.strip() if pub_date_elem is not None else ""

        pdf_url = f"{self.pdf_base_url}{paper_id}?pdf=render"
        if requests.get(pdf_url, timeout=10).status_code != 200:
            logger.info("No PDF found or access denied at %s",pdf_url)
            return {}

        return {
            "Title": title,
            "Authors": authors,
            "Abstract": abstract,
            "Publication Date": pub_date,
            "URL": pdf_url,
            "pdf_url": pdf_url,
            "filename": f"{paper_id}.pdf",
            "source": "pubmed",
            "pmc_id": paper_id,
        }

    def map_ids(self, input_id: str, map_url: str) -> str:
        """Mapping the PM ID or DOI of paper in order to get the PMC ID"""
        response = requests.get(f"{map_url}?ids={input_id}", timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        pmc_id = None
        if (
            (record := root.find("record")) is not None
            and record.attrib.get("pmcid")
            ):
            logger.info("Retrieved PMC ID for the given id %s", input_id)
            pmc_id =  record.attrib["pmcid"]
        if pmc_id is not None:
            logger.info("Found pmc id %s",pmc_id)
        return pmc_id

    def paper_retriever(
        self,
        paper_ids: List[str]
    ) ->dict[str, Any]:
        """
        Get metadata and PDF URL for an pubmed paper using its unique PMC ID.
        """
        logger.info("Fetching metadata from Pubmed for %s",paper_ids)
        self.load_hydra_configs()

        # Aggregate results
        article_data: dict[str, Any] = {}
        for pid in paper_ids:
            pre = pid.split(":")[0]
            pid = pid.split(":")[1]
            logger.info("Processing Pubmed ID: %s", pid)
            if pre == "pubmed":
                paper_id = self.map_ids(pid, self.map_url)
            else:
                paper_id = pid
            if paper_id is None:
                logger.info("PMC ID not available for %s",paper_id)
                continue
            # Fetch and parse metadata
            xml_root = self.fetch_metadata(self.metadata_url, paper_id)
            if xml_root is None:
                logger.info("No xml_root found for pubmed ID %s", paper_id)
                continue

            metadata = self.extract_metadata(
                xml_root, paper_id
            )
            article_data[paper_id] = metadata
            logger.info("Successfully fetched details for %s",paper_id)

        return {
            "article_data": article_data
        }
