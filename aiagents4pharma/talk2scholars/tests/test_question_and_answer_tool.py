"""
Unit tests for question_and_answer tool functionality.
"""

import unittest
from unittest.mock import MagicMock, patch


from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    generate_answer,
    question_and_answer_tool,
)


# Dummy document object for testing
class DummyDoc:
    """dummy document class to simulate PyPDFLoader output."""

    def __init__(self, content):
        """initialize with content."""
        self.page_content = content


# Dummy configuration objects to simulate Hydra config
class DummyQAConfig:
    """dummy configuration for question and answer tool."""

    chunk_size = 50
    chunk_overlap = 5
    num_retrievals = 2
    # No prompt_template provided


class DummyQAConfigWithPrompt:
    """Dummy configuration with custom prompt template."""

    chunk_size = 50
    chunk_overlap = 5
    num_retrievals = 2
    prompt_template = "Custom Prompt: {context} | Q: {question}"


class DummyArticleData(dict):
    """test class to simulate article_data."""

    def __bool__(self):
        """test if the object is truthy."""
        return True  # Make it truthy

    def keys(self):
        """test if the object has no keys."""
        return []  # Always return an empty list


# Dummy vector store implementation
class DummyVectorStore:
    """dummy vector store to simulate similarity search."""

    def __init__(self, documents, embedding_model):
        """dummy vector store initialization."""
        self.documents = documents

    def similarity_search(self, question, k):
        """simulate similarity search."""
        # Simply return the first k documents
        return self.documents[:k]


# Dummy LLM response and model
class DummyLLMResponse:
    """dummy LLM response."""

    def __init__(self, content):
        """initialize with content."""
        self.content = content


class DummyLLMModel:
    """dummy LLM model to simulate LLM invocation."""

    def invoke(self, prompt):
        """invoke the LLM with a prompt."""
        return DummyLLMResponse("LLM answer")


class TestGenerateAnswer(unittest.TestCase):
    """tests for the generate_answer function."""

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.initialize"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.InMemoryVectorStore"
    )
    def test_generate_answer_without_splitting(
        self, mock_vector_store, mock_PyPDFLoader, mock_hydra_init, mock_hydra_compose
    ):
        """
        Test generate_answer when the PDF loader returns a single document whose
        content is short (no splitting occurs) and no prompt_template is provided.
        """
        # Set up dummy config without prompt_template.
        dummy_cfg = DummyQAConfig()
        cfg_obj = type(
            "DummyHydraConfig",
            (),
            {"tools": type("DummyTools", (), {"question_and_answer": dummy_cfg})},
        )
        mock_hydra_compose.return_value = cfg_obj
        mock_hydra_init.return_value.__enter__.return_value = None

        # Simulate PDF loading: one document with content shorter than chunk_size.
        dummy_doc = DummyDoc("short content")
        loader_instance = MagicMock()
        loader_instance.load.return_value = [dummy_doc]
        mock_PyPDFLoader.return_value = loader_instance

        # Simulate vector store creation and similarity search.
        fake_vector_store = DummyVectorStore([dummy_doc], None)
        mock_vector_store.from_documents.return_value = fake_vector_store

        dummy_llm = DummyLLMModel()
        result = generate_answer(
            "What is it?", "http://dummy.pdf", MagicMock(), dummy_llm
        )

        # Since no prompt_template is provided, default prompt format is used.
        expected_prompt = f"Context:\n{dummy_doc.page_content}\n\nQuestion: What is it?"
        # We expect the dummy LLM to return "LLM answer"
        self.assertEqual(result, {"output_text": "LLM answer"})

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.initialize"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.CharacterTextSplitter"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.InMemoryVectorStore"
    )
    def test_generate_answer_with_splitting_and_custom_prompt(
        self,
        mock_vector_store,
        mock_text_splitter,
        mock_PyPDFLoader,
        mock_hydra_init,
        mock_hydra_compose,
    ):
        """
        Test generate_answer when splitting is triggered (multiple chunks) and a custom
        prompt_template is provided.
        """
        # Set up dummy config with prompt_template.
        dummy_cfg = DummyQAConfigWithPrompt()
        cfg_obj = type(
            "DummyHydraConfig",
            (),
            {"tools": type("DummyTools", (), {"question_and_answer": dummy_cfg})},
        )
        mock_hydra_compose.return_value = cfg_obj
        mock_hydra_init.return_value.__enter__.return_value = None

        # Simulate PDF loading: one document with content longer than chunk_size.
        long_content = "a" * 100  # 100 characters > chunk_size (50)
        dummy_doc = DummyDoc(long_content)
        loader_instance = MagicMock()
        loader_instance.load.return_value = [dummy_doc]
        mock_PyPDFLoader.return_value = loader_instance

        # Simulate text splitting: return two dummy chunks.
        splitted_docs = [DummyDoc("chunk1"), DummyDoc("chunk2")]
        splitter_instance = MagicMock()
        splitter_instance.split_documents.return_value = splitted_docs
        mock_text_splitter.return_value = splitter_instance

        # Simulate vector store: similarity search returns the splitted chunks.
        fake_vector_store = DummyVectorStore(splitted_docs, None)
        mock_vector_store.from_documents.return_value = fake_vector_store

        dummy_llm = DummyLLMModel()
        result = generate_answer(
            "What is the key point?", "http://dummy.pdf", MagicMock(), dummy_llm
        )

        # The context should be the concatenation of the two chunks.
        expected_context = "chunk1\n\nchunk2"
        expected_prompt = dummy_cfg.prompt_template.format(
            context=expected_context, question="What is the key point?"
        )
        self.assertEqual(result, {"output_text": "LLM answer"})

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.hydra.initialize"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.InMemoryVectorStore"
    )
    def test_generate_answer_without_splitting_long_doc_no_split(
        self, mock_vector_store, mock_PyPDFLoader, mock_hydra_init, mock_hydra_compose
    ):
        """
        Test generate_answer when the document does not trigger splitting because its
        content length is within chunk_size.
        """
        # Set up dummy config without prompt_template.
        dummy_cfg = DummyQAConfig()
        cfg_obj = type(
            "DummyHydraConfig",
            (),
            {"tools": type("DummyTools", (), {"question_and_answer": dummy_cfg})},
        )
        mock_hydra_compose.return_value = cfg_obj
        mock_hydra_init.return_value.__enter__.return_value = None

        # Simulate PDF loading: one document with content shorter than chunk_size.
        dummy_doc = DummyDoc("short content")
        loader_instance = MagicMock()
        loader_instance.load.return_value = [dummy_doc]
        mock_PyPDFLoader.return_value = loader_instance

        fake_vector_store = DummyVectorStore([dummy_doc], None)
        mock_vector_store.from_documents.return_value = fake_vector_store

        dummy_llm = DummyLLMModel()
        result = generate_answer(
            "What is this?", "http://dummy.pdf", MagicMock(), dummy_llm
        )
        self.assertEqual(result, {"output_text": "LLM answer"})


class TestQuestionAndAnswerToolSelection(unittest.TestCase):
    """tests for the question_and_answer_tool selection logic."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    def test_numeric_paper_selection(self, mock_generate_answer):
        """
        Test that a numeric reference in the question selects the correct paper.
        (Covers part of lines 174-213.)
        """
        mock_generate_answer.return_value = {"output_text": "Answer from paper2"}
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/paper1.pdf",
                    "Title": "First Paper",
                    "filename": "paper1.pdf",
                },
                "paper2": {
                    "pdf_url": "http://example.com/paper2.pdf",
                    "Title": "Second Paper",
                    "filename": "paper2.pdf",
                },
            },
        }
        tool_call_id = "test_numeric_selection"
        question = "Tell me about the 2nd paper."
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        result = question_and_answer_tool.run(tool_input)
        # generate_answer should be called with pdf_url from paper2.
        mock_generate_answer.assert_called_once_with(
            question,
            "http://example.com/paper2.pdf",
            state["text_embedding_model"],
            state["llm_model"],
        )
        output_message = result.update["messages"][0].content
        self.assertIn("paper2.pdf", output_message)
        self.assertIn("Answer from paper2", output_message)

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    def test_title_match_paper_selection(self, mock_generate_answer):
        """
        Test that when numeric reference is not found, the title matching logic selects
        the correct paper. (Covers part of lines 174-213.)
        """
        mock_generate_answer.return_value = {
            "output_text": "Answer from deep learning paper"
        }
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/paper1.pdf",
                    "Title": "Quantum Mechanics",
                    "filename": "paper1.pdf",
                },
                "paper2": {
                    "pdf_url": "http://example.com/paper2.pdf",
                    "Title": "Deep Learning Advances",
                    "filename": "paper2.pdf",
                },
            },
        }
        tool_call_id = "test_title_match"
        question = "What are the recent findings in deep learning?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        result = question_and_answer_tool.run(tool_input)
        # Title matching should select paper2.
        mock_generate_answer.assert_called_once_with(
            question,
            "http://example.com/paper2.pdf",
            state["text_embedding_model"],
            state["llm_model"],
        )
        output_message = result.update["messages"][0].content
        self.assertIn("paper2.pdf", output_message)
        self.assertIn("Answer from deep learning paper", output_message)

    def test_empty_article_data(self):
        """
        Test that when article_data exists but is empty (no paper keys), a ValueError is raised.
        """
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {},  # empty dict
        }
        tool_call_id = "test_empty_article_data"
        question = "What is the summary?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertEqual(str(context.exception), "No article_data found in state.")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    def test_generate_answer_exception(self, mock_generate_answer):
        """
        Test that if generate_answer raises an exception, the tool catches it and
        raises a ValueError with the appropriate error message.
        """
        mock_generate_answer.side_effect = Exception("Dummy error")
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/paper1.pdf",
                    "Title": "Test Paper",
                    "filename": "paper1.pdf",
                }
            },
        }
        tool_call_id = "test_generate_answer_exception"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertIn("Error processing PDF: Dummy error", str(context.exception))


class TestQuestionAndAnswerTool(unittest.TestCase):
    """tests for the question_and_answer_tool."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    def test_success(self, mock_generate_answer):
        """Test the successful generation of an answer using a valid state."""
        # Set up a fake answer response.
        mock_generate_answer.return_value = {"output_text": "This is the answer."}

        # Create a valid state with required models and article_data.
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/test.pdf",
                    "Title": "Test Paper",
                    "filename": "test.pdf",
                }
            },
        }
        tool_call_id = "test_call_1"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }

        # Run the tool.
        result = question_and_answer_tool.run(tool_input)
        update = result.update

        # Verify that generate_answer was called with the correct parameters.
        mock_generate_answer.assert_called_once_with(
            question,
            "http://example.com/test.pdf",
            state["text_embedding_model"],
            state["llm_model"],
        )

        # Check the output Command update.
        self.assertIn("messages", update)
        messages = update["messages"]
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)

        # Verify the ToolMessage content includes the expected answer text.
        message_content = messages[0].content
        self.assertIn("Answer based on PDF 'test.pdf':", message_content)
        self.assertIn("This is the answer.", message_content)

    def test_missing_text_embedding_model(self):
        """Test error when text_embedding_model is missing from state."""
        state = {
            # Missing text_embedding_model
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/test.pdf",
                    "Title": "Test Paper",
                }
            },
        }
        tool_call_id = "test_call_2"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertEqual(
            str(context.exception), "No text embedding model found in state."
        )

    def test_missing_llm_model(self):
        """Test error when llm_model is missing from state."""
        state = {
            "text_embedding_model": MagicMock(),
            # Missing llm_model
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/test.pdf",
                    "Title": "Test Paper",
                }
            },
        }
        tool_call_id = "test_call_3"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertEqual(str(context.exception), "No LLM model found in state.")

    def test_missing_article_data(self):
        """Test error when article_data is missing from state."""
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            # Missing article_data
        }
        tool_call_id = "test_call_4"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertEqual(str(context.exception), "No article_data found in state.")

    def test_missing_pdf_url(self):
        """Test error when the selected paper lacks a pdf_url."""
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": {
                "paper1": {
                    # pdf_url is missing
                    "Title": "Test Paper",
                    "filename": "test.pdf",
                }
            },
        }
        tool_call_id = "test_call_5"
        question = "What is the conclusion?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        expected_error = "No PDF URL found for selected paper: Test Paper"
        self.assertEqual(str(context.exception), expected_error)


class TestQuestionAndAnswerToolNoPapers(unittest.TestCase):
    """tests for the question_and_answer_tool when no papers are found."""

    def test_no_papers_found_in_article_data(self):
        """
        Test that when article_data is truthy but has no keys,
        a ValueError is raised with the expected error message.
        """
        state = {
            "text_embedding_model": MagicMock(),
            "llm_model": MagicMock(),
            "article_data": DummyArticleData(),  # Truthy but no keys
        }
        tool_call_id = "test_no_papers"
        question = "What is the summary?"
        tool_input = {
            "question": question,
            "tool_call_id": tool_call_id,
            "state": state,
        }
        with self.assertRaises(ValueError) as context:
            question_and_answer_tool.run(tool_input)
        self.assertEqual(str(context.exception), "No papers found in article_data.")
