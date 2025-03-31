"""
Unit tests for question_and_answer tool functionality.
"""

import contextlib
from unittest import mock

import hydra
import pytest
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    extract_text_from_pdf_data,
    generate_answer,
    question_and_answer_tool,
)

# pylint: disable=redefined-outer-name,too-few-public-methods
# pylint: disable=unused-argument,protected-access
# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-branches,too-many-statements
# pylint: disable=broad-exception-caught
# --- Dummy Hydra configuration and context ---


@contextlib.contextmanager
def dummy_hydra_initialize(*args, **kwargs):
    """dummy_hydra_initialize to replace the original one."""
    yield


def dummy_hydra_compose(*args, **kwargs):
    """dummy_hydra_compose to replace the original one."""

    class DummyQAConfig:
        """dummy QA config for testing."""

        chunk_size = 100
        chunk_overlap = 0
        num_retrievals = 1
        prompt_template = "Context:\n{context}\n\nQuestion: {question}"

    class DummyTools:
        """dummy tools for testing."""

        question_and_answer = DummyQAConfig()

    class DummyCfg:
        """dummy config for testing."""

        tools = DummyTools()

    return DummyCfg()


def setup_hydra(monkeypatch):
    """setup_hydra to replace the original one."""
    monkeypatch.setattr(hydra, "initialize", dummy_hydra_initialize)
    monkeypatch.setattr(hydra, "compose", dummy_hydra_compose)


DUMMY_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"%\xe2\xe3\xcf\xd3\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog /Pages 2 0 R >>\n"
    b"endobj\n"
    b"2 0 obj\n"
    b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>\n"
    b"endobj\n"
    b"3 0 obj\n"
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
    b"/Resources << /Font << /F1 5 0 R >> >> >>\n"
    b"endobj\n"
    b"4 0 obj\n"
    b"<< /Length 44 >>\n"
    b"stream\nBT\n/F1 24 Tf\n72 712 Td\n(Hello World) Tj\nET\nendstream\n"
    b"endobj\n"
    b"5 0 obj\n"
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
    b"endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n"
    b"0000000100 00000 n \n0000000150 00000 n \n0000000200 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n250\n%%EOF\n"
)


def test_extract_text_from_pdf_data():
    """
    Test that extract_text_from_pdf_data returns text containing 'Hello World'.
    """
    extracted_text = extract_text_from_pdf_data(DUMMY_PDF_BYTES)
    assert "Hello World" in extracted_text


class DummyVectorStore:
    """dummy vector store for testing."""

    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, question, k):
        """simulate similarity search."""
        return [Document(page_content="dummy context")]


class DummyEmbeddings:
    """simulate embeddings for testing."""

    def embed(self, text: str):
        """embed text into a vector."""
        # Dummy implementation.
        return [0.0] * len(text)


class DummyLLM:
    """simulate LLM for testing."""

    def invoke(self, prompt: str):
        """invoke LLM with a prompt."""

        # Always return a dummy answer.
        class DummyResponse:
            """dummy response for testing."""

            content = "dummy answer"

        return DummyResponse()


def setup_monkeypatch(monkeypatch):
    """setup_monkeypatch to replace the original one."""
    monkeypatch.setattr(hydra, "initialize", dummy_hydra_initialize)
    monkeypatch.setattr(hydra, "compose", dummy_hydra_compose)
    # Patch the vector store so that similarity_search returns a dummy document.

    monkeypatch.setattr(
        InMemoryVectorStore,
        "from_documents",
        lambda docs, model: DummyVectorStore(docs),
    )
    # Patch the PDF text extraction to avoid processing real PDF data.
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.extract_text_from_pdf_data",
        lambda pdf_bytes: "dummy text",
    )


def test_question_and_answer_tool_arxiv(monkeypatch):
    """extract text from arXiv PDF data and generate an answer."""
    setup_monkeypatch(monkeypatch)
    # Prepare a dummy state in arXiv format.
    dummy_pdf_data = {
        "pdf_object": b"dummy pdf bytes",
        "pdf_url": "http://example.com/dummy.pdf",
        "arxiv_id": "1234.5678",
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    result = question_and_answer_tool(
        {
            "question": "What is this PDF about?",
            "tool_call_id": "test-call-id-arxiv",
            "state": state,
        }
    )
    assert isinstance(result, Command)
    assert "messages" in result.update
    message = result.update["messages"][0]
    assert "ArXiv paper 1234.5678" in message.content
    assert "dummy answer" in message.content


def test_question_and_answer_tool_zotero(monkeypatch):
    """test extracting text from Zotero PDF data and generating an answer."""
    setup_monkeypatch(monkeypatch)
    # Prepare a dummy state in Zotero (nested) format.
    dummy_pdf_data = {
        "paper1": {
            "attachment1": {
                "data": b"dummy pdf bytes",
                "url": "",
                "filename": "dummy_zotero.pdf",
            }
        }
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    result = question_and_answer_tool(
        {
            "question": "Explain the content of the PDF",
            "tool_call_id": "test-call-id-zotero",
            "state": state,
        }
    )
    assert isinstance(result, Command)
    assert "messages" in result.update
    message = result.update["messages"][0]
    assert "dummy_zotero.pdf" in message.content
    assert "dummy answer" in message.content


def test_missing_text_embedding_model(monkeypatch):
    """test missing text embedding model in state."""
    setup_monkeypatch(monkeypatch)
    dummy_pdf_data = {
        "pdf_object": b"dummy pdf bytes",
        "pdf_url": "http://example.com/dummy.pdf",
        "arxiv_id": "1234.5678",
    }
    state = {
        "pdf_data": dummy_pdf_data,
        # Missing "text_embedding_model"
        "llm_model": DummyLLM(),
    }
    with pytest.raises(ValueError, match="No text embedding model found in state."):
        question_and_answer_tool(
            {
                "question": "What is this PDF about?",
                "tool_call_id": "test-call-id-missing-embeddings",
                "state": state,
            }
        )


def test_missing_llm_model(monkeypatch):
    """test missing LLM model in state."""
    setup_monkeypatch(monkeypatch)
    dummy_pdf_data = {
        "pdf_object": b"dummy pdf bytes",
        "pdf_url": "http://example.com/dummy.pdf",
        "arxiv_id": "1234.5678",
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        # Missing "llm_model"
    }
    with pytest.raises(ValueError, match="No LLM model found in state."):
        question_and_answer_tool(
            {
                "question": "What is this PDF about?",
                "tool_call_id": "test-call-id-missing-llm",
                "state": state,
            }
        )


def test_missing_pdf_data(monkeypatch):
    """test missing PDF data in state."""
    setup_monkeypatch(monkeypatch)
    state = {
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
        # Missing "pdf_data"
    }
    with pytest.raises(ValueError, match="No pdf_data found in state."):
        question_and_answer_tool(
            {
                "question": "What is this PDF about?",
                "tool_call_id": "test-call-id-missing-pdf",
                "state": state,
            }
        )


def test_generate_answer_bytes_success(monkeypatch):
    """test generating an answer from PDF bytes."""
    setup_hydra(monkeypatch)
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.extract_text_from_pdf_data",
        lambda pdf_bytes: "page1\npage2",
    )
    # Patch the vector store to return a dummy document.

    class DummyVectorStoreBytes:
        """dummy vector store for testing with bytes."""

        def __init__(self, documents):
            """initialize with documents."""
            self.documents = documents

        def similarity_search(self, question, k):
            """simulate similarity search."""
            return [Document(page_content="dummy context bytes")]

    monkeypatch.setattr(
        InMemoryVectorStore,
        "from_documents",
        lambda docs, model: DummyVectorStoreBytes(docs),
    )

    result = generate_answer(
        "dummy question", b"dummy pdf bytes", DummyEmbeddings(), DummyLLM()
    )
    assert "dummy answer" in result["output_text"]


def test_generate_answer_bytes_exception(monkeypatch):
    """test generating an answer from PDF bytes with exception handling."""
    setup_hydra(monkeypatch)

    def failing_extract(pdf_bytes):
        """dummy function to simulate a failure in text extraction."""
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.extract_text_from_pdf_data",
        failing_extract,
    )

    # Patch PyPDFLoader to return a dummy document list.
    class DummyLoader:
        """dummy loader for testing."""

        def __init__(self, path):
            """initialize with path."""
            self.path = path

        def load(self):
            """load dummy document."""
            return [Document(page_content="loaded via PyPDFLoader")]

    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader",
        DummyLoader,
    )

    class DummyVectorStoreFallback:
        """dummy vector store for testing with fallback."""

        def __init__(self, documents):
            """initialize with documents."""
            self.documents = documents

        def similarity_search(self, question, k):
            """simulate similarity search."""
            return [Document(page_content="dummy context fallback")]

    monkeypatch.setattr(
        InMemoryVectorStore,
        "from_documents",
        lambda docs, model: DummyVectorStoreFallback(docs),
    )

    result = generate_answer(
        "dummy question", b"dummy pdf bytes", DummyEmbeddings(), DummyLLM()
    )
    assert "dummy answer" in result["output_text"]


def test_generate_answer_url(monkeypatch):
    """test generating an answer from a PDF URL."""
    setup_hydra(monkeypatch)

    class DummyLoaderURL:
        """dummy loader for testing with URL."""

        def __init__(self, url):
            """initialize with URL."""
            self.url = url

        def load(self):
            """load dummy document."""
            return [Document(page_content="loaded via URL")]

    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader",
        DummyLoaderURL,
    )

    class DummyVectorStoreURL:
        """dummy vector store for testing with URL."""

        def __init__(self, documents):
            """initialize with documents."""
            self.documents = documents

        def similarity_search(self, question, k):
            """simulate similarity search."""
            return [Document(page_content="dummy context URL")]

    monkeypatch.setattr(
        InMemoryVectorStore,
        "from_documents",
        lambda docs, model: DummyVectorStoreURL(docs),
    )

    result = generate_answer(
        "dummy question", "http://dummy.url/dummy.pdf", DummyEmbeddings(), DummyLLM()
    )
    assert "dummy answer" in result["output_text"]


def test_generate_answer_invalid_source(monkeypatch):
    """test generating an answer with an invalid PDF source."""
    with pytest.raises(ValueError, match="pdf_source must be either bytes"):
        generate_answer("dummy question", 12345, DummyEmbeddings(), DummyLLM())


def test_question_and_answer_tool_zotero_multiple(monkeypatch):
    """test extracting text from multiple Zotero PDF data and generating an answer."""
    setup_monkeypatch(monkeypatch)
    dummy_pdf_data = {
        "paper1": {
            "attachment1": {
                "data": b"dummy pdf bytes paper1",
                "url": "",
                "filename": "Deep Learning Advances.pdf",
            }
        },
        "paper2": {
            "attachment1": {
                "data": b"dummy pdf bytes paper2",
                "url": "",
                "filename": "Quantum Computing Insights.pdf",
            }
        },
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    result = question_and_answer_tool(
        {
            "question": "What are the insights on Quantum Computing?",
            "tool_call_id": "test-call-id-multi-zotero",
            "state": state,
        }
    )
    assert "Quantum Computing Insights.pdf" in result.update["messages"][0].content


def test_question_and_answer_tool_generate_answer_exception(monkeypatch):
    """test generating an answer with an exception in the generate_answer function."""

    def failing_generate_answer(*args, **kwargs):
        raise RuntimeError("simulated generate_answer failure")

    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer",
        failing_generate_answer,
    )
    dummy_pdf_data = {
        "pdf_object": b"dummy pdf bytes",
        "pdf_url": "http://example.com/dummy.pdf",
        "arxiv_id": "1234.5678",
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    with pytest.raises(
        ValueError, match="Error processing PDF: simulated generate_answer failure"
    ):
        question_and_answer_tool(
            {
                "question": "What is this PDF about?",
                "tool_call_id": "test-call-id-gen-err",
                "state": state,
            }
        )


class NonEmptyButNoKeys(dict):
    """non-empty dict with no keys."""

    def __bool__(self):
        """bool method to always return True."""
        return True

    def keys(self):
        """keys method to return an empty list."""
        return []


def test_question_and_answer_tool_empty_papers(monkeypatch):
    """test empty papers in PDF data."""
    setup_monkeypatch(monkeypatch)
    state = {
        "pdf_data": NonEmptyButNoKeys(),
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    with pytest.raises(ValueError, match="No papers with PDFs found in pdf_data."):
        question_and_answer_tool(
            {
                "question": "dummy question",
                "tool_call_id": "test-empty-papers",
                "state": state,
            }
        )


def test_question_and_answer_tool_no_attachments(monkeypatch):
    """test no attachments in PDF data."""
    setup_monkeypatch(monkeypatch)
    dummy_pdf_data = {"paper1": {}}  # Empty attachments dict.
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    with pytest.raises(ValueError, match="No PDF attachments found for paper paper1."):
        question_and_answer_tool(
            {
                "question": "dummy question",
                "tool_call_id": "test-no-attachments",
                "state": state,
            }
        )


def test_question_and_answer_tool_no_pdf_data(monkeypatch):
    """test no PDF data in state."""
    setup_monkeypatch(monkeypatch)
    dummy_pdf_data = {
        "pdf_object": None,
        "pdf_url": "",
        "arxiv_id": "1234.5678",
    }
    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }
    with pytest.raises(
        ValueError, match="Neither PDF binary data nor URL is available."
    ):
        question_and_answer_tool(
            {
                "question": "dummy question",
                "tool_call_id": "test-no-pdf",
                "state": state,
            }
        )


def test_generate_answer_default_prompt(monkeypatch):
    """test generating an answer with default prompt."""

    def dummy_hydra_compose_no_prompt(*args, **kwargs):
        """dummy_hydra_compose_no_prompt to replace the original one."""

        class DummyQAConfig:
            """dummy QA config for testing."""

            chunk_size = 100
            chunk_overlap = 0
            num_retrievals = 1

        class DummyTools:
            """dummy tools for testing."""

            question_and_answer = DummyQAConfig()

        class DummyCfg:
            """dummy config for testing."""

            tools = DummyTools()

        return DummyCfg()

    monkeypatch.setattr(hydra, "compose", dummy_hydra_compose_no_prompt)
    monkeypatch.setattr(hydra, "initialize", dummy_hydra_initialize)

    class DummyVectorStoreNoPrompt:
        """dummy vector store for testing with no prompt."""

        def __init__(self, documents):
            """initialize with documents."""
            self.documents = documents

        def similarity_search(self, question, k):
            """simulate similarity search."""
            return [Document(page_content="default context")]

    monkeypatch.setattr(
        InMemoryVectorStore,
        "from_documents",
        lambda docs, model: DummyVectorStoreNoPrompt(docs),
    )

    # Patch text extraction and text splitting.
    monkeypatch.setattr(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.extract_text_from_pdf_data",
        lambda pdf: "dummy text",
    )

    monkeypatch.setattr(CharacterTextSplitter, "split_text", lambda self, text: [text])

    class DummyLLM:
        """dummy LLM for testing."""

        def invoke(self, prompt: str):
            """invoke LLM with a prompt."""

            class DummyResponse:
                """dummy response for testing."""

                content = prompt

            return DummyResponse()

    dummy_pdf_source = b"dummy pdf bytes"
    result = generate_answer(
        "What is the question?", dummy_pdf_source, DummyEmbeddings(), DummyLLM()
    )
    expected_prompt = "Context:\ndefault context\n\nQuestion: What is the question?"
    assert expected_prompt in result["output_text"]


def test_dummy_embeddings_embed():
    """test DummyEmbeddings embed method."""
    dummy = DummyEmbeddings()
    sample_text = "hello"
    expected_output = [0.0] * len(sample_text)
    assert dummy.embed(sample_text) == expected_output


def test_question_and_answer_tool_dash_title_extraction(monkeypatch):
    """Test extracting title from filenames with dashes (covers line 257)."""
    setup_monkeypatch(monkeypatch)

    # Create test data with filenames containing dashes
    dummy_pdf_data = {
        "paper1": {
            "attachment1": {
                "data": b"dummy pdf bytes",
                "url": "",
                "filename": "Author - 2020 - Complex Title With Multiple Words.pdf",
            }
        },
        "paper2": {
            "attachment1": {
                "data": b"dummy pdf bytes",
                "url": "",
                "filename": "No Dashes Here.pdf",
            }
        },
    }

    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }

    # Test with a query matching words from the title part (after the last dash)
    result = question_and_answer_tool(
        {
            "question": "Can you explain the Complex Title With Multiple Words paper?",
            "tool_call_id": "test-dash-title",
            "state": state,
        }
    )

    # Verify that the paper with the matching title was selected
    assert (
        "Author - 2020 - Complex Title With Multiple Words.pdf"
        in result.update["messages"][0].content
    )


def test_question_and_answer_tool_numeric_reference(monkeypatch):
    """Test selecting a paper by numeric reference with improved regex pattern."""
    setup_monkeypatch(monkeypatch)

    # Create test data with ordered, predictable keys
    dummy_pdf_data = {
        "key1": {
            "attachment1": {
                "data": b"dummy pdf bytes paper1",
                "url": "",
                "filename": "Paper One.pdf",
            }
        },
        "key2": {
            "attachment1": {
                "data": b"dummy pdf bytes paper2",
                "url": "",
                "filename": "Paper Two.pdf",
            }
        },
        "key3": {
            "attachment1": {
                "data": b"dummy pdf bytes paper3",
                "url": "",
                "filename": "Paper Three.pdf",
            }
        },
    }

    state = {
        "pdf_data": dummy_pdf_data,
        "text_embedding_model": DummyEmbeddings(),
        "llm_model": DummyLLM(),
    }

    # Test cases for various numeric patterns
    test_cases = [
        # Test both "paper number" and "number paper" formats
        ("show me paper 2", 2, "Paper Two.pdf"),
        ("get the 3rd paper", 3, "Paper Three.pdf"),
        ("paper 1 summary", 1, "Paper One.pdf"),
    ]

    for query, expected_num, expected_filename in test_cases:
        # Use mock.patch to observe the info logging
        with mock.patch("logging.Logger.info") as mock_info:
            # Run the function with the test query
            result = question_and_answer_tool(
                {
                    "question": query,
                    "tool_call_id": "test-numeric-reference",
                    "state": state,
                }
            )

            # Check that the numeric selection log message was generated
            expected_log = f"Selected paper {expected_num} based on numerical reference"
            log_messages = [
                call.args[0] % call.args[1:] for call in mock_info.call_args_list
            ]

            assert any(
                expected_log in message for message in log_messages
            ), f"Expected log message '{expected_log}' not found"

            # Verify the correct paper was selected
            assert (
                expected_filename in result.update["messages"][0].content
            ), f"Expected '{expected_filename}' to be in the response content"
