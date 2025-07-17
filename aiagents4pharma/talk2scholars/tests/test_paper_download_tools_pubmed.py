"""
Unit tests for pubmed paper downloading functionality, including:
- download_pubmed_paper tool function.
"""
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

from aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_input import (
    DownloadPubmedPaperInput)

@patch(
"aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_input.hydra.initialize"
)
@patch(
"aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_input.hydra.compose"
)
def test_load_hydra_configs_runs_and_sets_attributes(mock_compose, mock_initialize):
    """
    Ensures:
    - logger.info runs
    - hydra.initialize runs
    - hydra.compose runs
    - self.metadata_url etc. get set
    """

    # Fake config structure
    mock_cfg = MagicMock()
    mock_cfg.tools.download_pubmed_paper.metadata_url = "https://metadata"
    mock_cfg.tools.download_pubmed_paper.pdf_base_url = "https://pdf"
    mock_cfg.tools.download_pubmed_paper.map_url = "https://map"
    mock_compose.return_value = mock_cfg

    retriever = DownloadPubmedPaperInput()
    retriever.load_hydra_configs()

    mock_initialize.assert_called_once_with(version_base=None, config_path="../../configs")
    mock_compose.assert_called_once_with(
        config_name="config",
        overrides=["tools/download_pubmed_paper=default"]
    )

    assert retriever.metadata_url == "https://metadata"
    assert retriever.pdf_base_url == "https://pdf"
    assert retriever.map_url == "https://map"

@patch("requests.get")
def test_fetch_metadata_success(mock_get):
    """Pubmed happy run"""
    # Covers fetch_metadata (lines 27–34)
    mock_response = MagicMock()
    mock_response.text = "<foo>bar</foo>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    retriever = DownloadPubmedPaperInput()
    result = retriever.fetch_metadata("http://example.org", "123")
    assert "data" in result
    assert isinstance(result["data"], ET.Element)
    mock_get.assert_called_once_with(
        "http://example.org",
        params={"db": "pmc", "id": "123", "retmode": "xml"},
        timeout=10,
    )


@patch("requests.get")
def test_extract_metadata_success_pdf_found(mock_get):
    """Covers extract_metadata success path (lines 42 to 49)"""
    retriever = DownloadPubmedPaperInput()
    retriever.pdf_base_url = "http://example.org/"

    xml = ET.fromstring("""
    <article>
        <front>
            <article-meta>
                <title-group>
                    <article-title>Example Title</article-title>
                </title-group>
                <abstract><p>Example abstract text.</p></abstract>
                <contrib-group>
                    <contrib contrib-type="author">
                        <name>
                            <surname>Smith</surname>
                            <given-names>Jane</given-names>
                        </name>
                    </contrib>
                </contrib-group>
                <pub-date><year>2024</year></pub-date>
            </article-meta>
        </front>
    </article>
    """)
    mock_get.return_value.status_code = 200

    result = retriever.extract_metadata({"data": xml}, "PMC123456")
    assert result["Title"] == "Example Title"
    assert "Jane Smith" in result["Authors"]
    assert "Example abstract text." in result["Abstract"]
    assert result["pdf_url"].startswith("http://example.org/PMC123456")


@patch("requests.get")
def test_extract_metadata_no_pdf_found(mock_get):
    """Covers extract_metadata when PDF URL not found (lines 42to49)"""
    retriever = DownloadPubmedPaperInput()
    retriever.pdf_base_url = "http://example.org/"
    xml = ET.fromstring("""
    <article><front><article-meta><title-group><article-title>Title</article-title></title-group>
    </article-meta></front></article>
    """)
    mock_get.return_value.status_code = 404  # PDF not found

    result = retriever.extract_metadata({"data": xml}, "PMC123456")
    assert not result

@patch("requests.get")
def test_map_ids_success(mock_get):
    """Map ids success"""
    retriever = DownloadPubmedPaperInput()
    mock_response = MagicMock()
    mock_response.text = """<eSummaryResult>
        <record pmcid="PMC123456"></record>
    </eSummaryResult>"""
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    pmc_id = retriever.map_ids("98765", "http://example.org/efetch")
    assert pmc_id == "PMC123456"
    mock_get.assert_called_once_with(
        "http://example.org/efetch?ids=98765",
        timeout=10
    )


@patch.object(DownloadPubmedPaperInput, "load_hydra_configs")
@patch.object(DownloadPubmedPaperInput, "map_ids")
@patch.object(DownloadPubmedPaperInput, "fetch_metadata")
@patch.object(DownloadPubmedPaperInput, "extract_metadata")
def test_paper_retriever_xml_root_none(
    mock_extract, mock_fetch, mock_map, mock_load
):
    """Fetch data returns none"""
    # Covers lines 127–129: skips if fetch_metadata returns None
    mock_load.return_value = None
    mock_map.return_value = "PMC123456"
    mock_fetch.return_value = None  # Simulate XML fetch fail
    mock_extract.return_value = {"dummy": "data"}

    retriever = DownloadPubmedPaperInput()
    result = retriever.paper_retriever(["pubmed:98765"])
    assert not result["article_data"]
    mock_extract.assert_not_called()

@patch.object(DownloadPubmedPaperInput, "load_hydra_configs")
@patch.object(DownloadPubmedPaperInput, "map_ids")
@patch.object(DownloadPubmedPaperInput, "fetch_metadata")
@patch.object(DownloadPubmedPaperInput, "extract_metadata")
def test_paper_retriever_pubmed_branch_maps_id_and_extracts(
    mock_extract, mock_fetch, mock_map, mock_load
):
    """
    Covers:
      - `if pre == "pubmed"` -> map_ids() called.
      - extract_metadata is called.
      - article_data populated.
    """
    mock_load.return_value = None
    mock_map.return_value = "PMC123456"
    mock_fetch.return_value = {"data": "xml_data"}
    mock_extract.return_value = {"Title": "My Paper"}

    retriever = DownloadPubmedPaperInput()
    result = retriever.paper_retriever(["pubmed:98765"])
    assert "PMC123456" in result["article_data"]
    assert result["article_data"]["PMC123456"]["Title"] == "My Paper"
    mock_map.assert_called_once_with("98765", retriever.map_url)
    mock_fetch.assert_called_once_with(retriever.metadata_url, "PMC123456")
    mock_extract.assert_called_once_with({"data": "xml_data"}, "PMC123456")


@patch.object(DownloadPubmedPaperInput, "load_hydra_configs")
@patch.object(DownloadPubmedPaperInput, "fetch_metadata")
@patch.object(DownloadPubmedPaperInput, "extract_metadata")
def test_paper_retriever_else_branch_uses_pid_directly_and_extracts(
    mock_extract, mock_fetch, mock_load
):
    """
    Covers:
      - `else` branch when pre != pubmed.
      - Does not call map_ids.
      - Uses pid directly.
      - extract_metadata is called.
    """
    mock_load.return_value = None
    mock_fetch.return_value = {"data": "xml_data"}
    mock_extract.return_value = {"Title": "Non-PubMed Paper"}

    retriever = DownloadPubmedPaperInput()
    result = retriever.paper_retriever(["pmcid:PMC123456"])
    assert "PMC123456" in result["article_data"]
    assert result["article_data"]["PMC123456"]["Title"] == "Non-PubMed Paper"
    mock_fetch.assert_called_once_with(retriever.metadata_url, "PMC123456")
    mock_extract.assert_called_once_with({"data": "xml_data"}, "PMC123456")


@patch.object(DownloadPubmedPaperInput, "load_hydra_configs")
@patch.object(DownloadPubmedPaperInput, "map_ids")
@patch.object(DownloadPubmedPaperInput, "fetch_metadata")
@patch.object(DownloadPubmedPaperInput, "extract_metadata")
def test_paper_retriever_skips_when_paper_id_none(
    mock_extract, mock_fetch, mock_map, mock_load
):
    """
    Covers:
      - `if paper_id == None` skip.
      - extract_metadata not called.
    """
    mock_load.return_value = None
    mock_map.return_value = None  # Simulate map_ids failed.
    retriever = DownloadPubmedPaperInput()
    result = retriever.paper_retriever(["pubmed:98765"])

    assert not result["article_data"]
    mock_map.assert_called_once_with("98765", retriever.map_url)
    mock_fetch.assert_not_called()
    mock_extract.assert_not_called()
