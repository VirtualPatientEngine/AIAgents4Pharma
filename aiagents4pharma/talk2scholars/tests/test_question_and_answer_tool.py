"""
Unit tests for question_and_answer tool functionality.
"""

from langchain.docstore.document import Document

from ..tools.pdf import question_and_answer
from ..tools.pdf.question_and_answer import (
    extract_text_from_pdf_data,
    question_and_answer_tool,
    generate_answer,
)


def test_extract_text_from_pdf_data():
    """
    Test that extract_text_from_pdf_data returns text containing 'Hello World'.
    """
    extracted_text = extract_text_from_pdf_data(DUMMY_PDF_BYTES)
    assert "Hello World" in extracted_text


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


def fake_generate_answer(question, pdf_bytes, _llm_model):
    """
    Fake generate_answer function to bypass external dependencies.
    """
    return {
        "answer": "Mock answer",
        "question": question,
        "pdf_bytes_length": len(pdf_bytes),
    }


def test_question_and_answer_tool_success(monkeypatch):
    """
    Test that question_and_answer_tool returns the expected result on success.
    """
    monkeypatch.setattr(
        question_and_answer, "generate_answer", fake_generate_answer
    )
    # Create a valid state with pdf_data containing both pdf_object and pdf_url,
    # and include a dummy llm_model.
    state = {
        "pdf_data": {"pdf_object": DUMMY_PDF_BYTES, "pdf_url": "http://dummy.url"},
        "llm_model": object(),  # Provide a dummy LLM model instance.
    }
    question = "What is in the PDF?"
    # Call the underlying function directly via .func to bypass the StructuredTool wrapper.
    result = question_and_answer_tool.func(
        question=question, tool_call_id="test_call_id", state=state
    )
    assert result["answer"] == "Mock answer"
    assert result["question"] == question
    assert result["pdf_bytes_length"] == len(DUMMY_PDF_BYTES)


def test_question_and_answer_tool_no_pdf_data():
    """
    Test that an error is returned if the state lacks the 'pdf_data' key.
    """
    state = {}  # pdf_data key is missing.
    question = "Any question?"
    result = question_and_answer_tool.func(
        question=question, tool_call_id="test_call_id", state=state
    )
    messages = result.update["messages"]
    assert any("No pdf_data found in state." in msg.content for msg in messages)


def test_question_and_answer_tool_no_pdf_object():
    """
    Test that an error is returned if the pdf_object is missing within pdf_data.
    """
    state = {"pdf_data": {"pdf_object": None}}
    question = "Any question?"
    result = question_and_answer_tool.func(
        question=question, tool_call_id="test_call_id", state=state
    )
    messages = result.update["messages"]
    assert any(
        "PDF binary data is missing in the pdf_data from state." in msg.content
        for msg in messages
    )


def test_question_and_answer_tool_no_llm_model():
    """
    Test that an error is returned if the LLM model is missing in the state.
    """
    state = {
        "pdf_data": {"pdf_object": DUMMY_PDF_BYTES, "pdf_url": "http://dummy.url"}
        # Note: llm_model is intentionally omitted.
    }
    question = "What is in the PDF?"
    result = question_and_answer_tool.func(
        question=question, tool_call_id="test_call_id", state=state
    )
    assert result == {"error": "No LLM model found in state."}


def test_generate_answer(monkeypatch):
    """
    Test generate_answer function with controlled monkeypatched dependencies.
    """

    def fake_split_text(_self, _text):
        """Fake split_text method that returns controlled chunks."""
        return ["chunk1", "chunk2"]

    monkeypatch.setattr(
        question_and_answer.CharacterTextSplitter, "split_text", fake_split_text
    )

    def fake_annoy_from_documents(_documents, _embeddings):
        """
        Fake Annoy.from_documents function that returns a fake vector store.
        """
        # pylint: disable=too-few-public-methods, unused-argument
        class FakeVectorStore:
            """Fake vector store for similarity search."""
            def similarity_search(self, _question, k):
                """Return a list with a single dummy Document."""
                return [Document(page_content="dummy content")]
        return FakeVectorStore()

    monkeypatch.setattr(
        question_and_answer.Annoy, "from_documents", fake_annoy_from_documents
    )

    def fake_load_qa_chain(_llm, chain_type):  # chain_type matches the keyword argument
        """
        Fake load_qa_chain function that returns a fake QA chain.
        """
        # pylint: disable=too-few-public-methods, unused-argument
        class FakeChain:
            """Fake QA chain for testing generate_answer."""
            def invoke(self, **kwargs):
                """
                Fake invoke method that returns a mock answer.
                """
                input_data = kwargs.get("input")
                return {
                    "answer": "real mock answer",
                    "question": input_data.get("question"),
                }
        return FakeChain()

    monkeypatch.setattr(question_and_answer, "load_qa_chain", fake_load_qa_chain)
    # Set dummy configuration values so that generate_answer can run.
    question_and_answer.cfg.chunk_size = 1000
    question_and_answer.cfg.chunk_overlap = 0
    question_and_answer.cfg.openai_api_key = "dummy_key"
    question_and_answer.cfg.num_retrievals = 1
    question_and_answer.cfg.qa_chain_type = "dummy-chain"

    question = "What is in the PDF?"
    dummy_llm_model = object()  # A dummy model placeholder.
    answer = generate_answer(question, DUMMY_PDF_BYTES, dummy_llm_model)
    assert answer["answer"] == "real mock answer"
    assert answer["question"] == question
