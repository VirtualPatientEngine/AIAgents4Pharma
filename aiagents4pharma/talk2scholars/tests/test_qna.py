import io
import pytest
from ..tools.pdf import qna
from ..tools.pdf.qna import extract_text_from_pdf_data, qna_tool, generate_answer
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from io import BytesIO
from langchain_core.messages import ToolMessage
from langgraph.types import Command


# A minimal valid PDF binary that includes "Hello World".
dummy_pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n" \
                  b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n" \
                  b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n" \
                  b"4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n72 712 Td\n(Hello World) Tj\nET\nendstream\nendobj\n" \
                  b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n" \
                  b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n" \
                  b"0000000100 00000 n \n0000000150 00000 n \n0000000200 00000 n \n" \
                  b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n250\n%%EOF\n"

# Unit test for extract_text_from_pdf_data.
def test_extract_text_from_pdf_data():
    extracted_text = extract_text_from_pdf_data(dummy_pdf_bytes)
    # Check that the known text "Hello World" is present.
    assert "Hello World" in extracted_text

# Fake generate_answer to bypass external dependencies in integration tests.
def fake_generate_answer(question, pdf_bytes):
    return {
        "answer": "Mock answer",
        "question": question,
        "pdf_bytes_length": len(pdf_bytes)
    }

# Integration test for a successful run of qna_tool.
def test_qna_tool_success(monkeypatch):
    # Monkeypatch generate_answer to avoid real external calls.
    monkeypatch.setattr(qna, "generate_answer", fake_generate_answer)
    
    # Create a valid state with pdf_data containing both pdf_object and pdf_url.
    state = {"pdf_data": {"pdf_object": dummy_pdf_bytes, "pdf_url": "http://dummy.url"}}
    question = "What is in the PDF?"
    
    # Call the underlying function directly via .func to bypass StructuredTool wrapper.
    result = qna_tool.func(question=question, tool_call_id="test_call_id", state=state)
    assert result["answer"] == "Mock answer"
    assert result["question"] == question
    assert result["pdf_bytes_length"] == len(dummy_pdf_bytes)

# Integration test for error when pdf_data is missing in state.
def test_qna_tool_no_pdf_data():
    state = {}  # pdf_data key is missing.
    question = "Any question?"
    
    result = qna_tool.func(question=question, tool_call_id="test_call_id", state=state)
    # Access the Command object's update attribute.
    messages = result.update["messages"]
    assert any("No pdf_data found in state." in msg.content for msg in messages)

# Integration test for error when pdf_object is missing within pdf_data.
def test_qna_tool_no_pdf_object():
    # Here, we provide pdf_data with pdf_object explicitly set to None.
    state = {"pdf_data": {"pdf_object": None}}
    question = "Any question?"
    
    result = qna_tool.func(question=question, tool_call_id="test_call_id", state=state)
    messages = result.update["messages"]
    assert any("PDF binary data is missing in the pdf_data from state." in msg.content for msg in messages)
