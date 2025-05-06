"""
Unit tests for question_and_answer tool functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    Vectorstore,
    generate_answer,
    question_and_answer,
)


class TestQuestionAndAnswerTool(unittest.TestCase):
    """tests for question_and_answer tool functionality."""

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    def test_add_paper(self, mock_pypdf_loader):
        """test adding a paper to the vector store."""
        # Mock the PDF loader
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [Document(page_content="Page content")]

        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Add a paper
        vector_store.add_paper(
            paper_id="test_paper",
            pdf_url="http://example.com/test.pdf",
            paper_metadata={"Title": "Test Paper"},
        )

        # Check if the paper was added
        self.assertIn("test_paper_0", vector_store.documents)

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    def test_add_paper_already_loaded(self, mock_pypdf_loader):
        """Test that adding a paper that is already loaded does not re-load or add new documents."""
        # Mock the PDF loader (it should not be used when the paper is already loaded)
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [Document(page_content="Page content")]

        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Simulate the paper already being loaded.
        vector_store.loaded_papers.add("test_paper")
        # Capture the initial state of documents (should be empty)
        initial_documents = dict(vector_store.documents)

        # Attempt to add the same paper again.
        vector_store.add_paper(
            paper_id="test_paper",
            pdf_url="http://example.com/test.pdf",
            paper_metadata={"Title": "Test Paper"},
        )

        # Verify that no new paper was added by checking:
        # 1. The loaded papers set remains unchanged.
        self.assertEqual(vector_store.loaded_papers, {"test_paper"})
        # 2. The documents dictionary remains unchanged.
        self.assertEqual(vector_store.documents, initial_documents)
        # 3. The PDF loader was not called at all.
        mock_loader.load.assert_not_called()

    def test_build_vector_store(self):
        """test building the vector store."""
        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Add a mock document
        vector_store.documents["test_doc"] = Document(page_content="Test content")

        # Mock the embed_documents method to return a list of embeddings
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        # Build vector store
        vector_store.build_vector_store()

        # Check if the vector store is built
        self.assertIsNotNone(vector_store.vector_store)

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_hydra_config"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.NVIDIARerank")
    def test_rank_papers_by_query(self, mock_nvidia_rerank, mock_load_config):
        """test ranking papers by query."""
        # Create a mock config object with attributes
        mock_config = MagicMock()
        mock_config.reranker.model = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        mock_config.reranker.api_key = "dummy_api_key"

        # Patch load_hydra_config to return the mock config object
        mock_load_config.return_value = mock_config

        # Mock the re-ranker instance.
        mock_reranker = mock_nvidia_rerank.return_value
        mock_reranker.compress_documents.return_value = [
            Document(
                page_content="Aggregated content", metadata={"paper_id": "test_paper"}
            )
        ]

        # Create a mock embedding model.
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore.
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Add a mock document.
        vector_store.documents["test_doc"] = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )

        # Rank papers.
        ranked_papers = vector_store.rank_papers_by_query(query="test query")

        # Check if the ranking is correct (updated expectation: a list of paper IDs)
        self.assertEqual(ranked_papers[0], "test_paper")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.maximal_marginal_relevance"
    )
    def test_retrieve_relevant_chunks(self, mock_mmr):
        """Test retrieving relevant chunks without filters."""
        mock_mmr.return_value = [0]
        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        vector_store = Vectorstore(embedding_model=mock_embedding_model)
        vector_store.vector_store = True
        vector_store.documents["test_doc"] = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )

        results = vector_store.retrieve_relevant_chunks(query="test query")
        assert len(results) == 1
        assert results[0].metadata["paper_id"] == "test_paper"

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.BaseChatModel")
    def test_generate_answer(self, mock_base_chat_model):
        """test generating an answer."""
        # Mock the language model
        mock_llm = mock_base_chat_model.return_value
        mock_llm.invoke.return_value.content = "Generated answer"

        # Create a mock document
        mock_document = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )

        # Generate answer
        result = generate_answer(
            question="What is the test?",
            retrieved_chunks=[mock_document],
            llm_model=mock_llm,
        )

        # Check if the answer is generated correctly
        self.assertEqual(result["output_text"], "Generated answer")

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    def test_add_paper_exception_handling(self, mock_pypdf_loader):
        """Test exception handling when adding a paper."""
        # Mock the PDF loader to raise an exception.
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.side_effect = Exception("Loading error")

        # Mock embedding model.
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore.
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Attempt to add a paper and expect an exception.
        with self.assertRaises(Exception) as context:
            vector_store.add_paper(
                paper_id="test_paper",
                pdf_url="http://example.com/test.pdf",
                paper_metadata={"Title": "Test Paper"},
            )

        # Verify that the exception message is as expected.
        self.assertEqual(str(context.exception), "Loading error")

    def test_build_vector_store_no_documents(self):
        """Test building vector store with no documents results in an unchanged vector_store."""
        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore without adding any documents
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Attempt to build vector store
        vector_store.build_vector_store()

        # Instead of checking log output, check that vector_store remains None (or unchanged)
        self.assertIsNone(vector_store.vector_store)

    def test_build_vector_store_already_built(self):
        """Test that calling build_vector_store when
        it is already built does not change the store."""
        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Add a mock document
        vector_store.documents["test_doc"] = Document(page_content="Test content")

        # Mock the embed_documents method to return a list of embeddings
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        # Build vector store once
        vector_store.build_vector_store()
        first_build = vector_store.vector_store

        # Attempt to build vector store again
        vector_store.build_vector_store()

        # Check that the vector store remains unchanged (i.e. same object/state)
        self.assertEqual(vector_store.vector_store, first_build)

    def test_retrieve_relevant_chunks_vector_store_not_built(self):
        """Test retrieving relevant chunks when the vector store is not built."""
        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore without adding any documents
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Attempt to retrieve relevant chunks (vector_store.vector_store is None)
        result = vector_store.retrieve_relevant_chunks(query="test query")

        # Verify that an empty list is returned since the vector store is not built.
        self.assertEqual(result, [])

    def test_retrieve_relevant_chunks_with_paper_ids(self):
        """Test retrieving relevant chunks with specific paper_ids when the store is not built."""
        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)
        # Mock embed_documents method to return embeddings of fixed length
        mock_embedding_model.embed_documents.return_value = [MagicMock()] * 2

        # Initialize Vectorstore and add documents
        vector_store = Vectorstore(embedding_model=mock_embedding_model)
        vector_store.documents = {
            "doc1": Document(page_content="content1", metadata={"paper_id": "paper1"}),
            "doc2": Document(page_content="content2", metadata={"paper_id": "paper2"}),
        }

        # Leave vector_store.vector_store as None to trigger the branch that returns an empty list
        vector_store.vector_store = None

        # Call retrieve_relevant_chunks with specific paper_ids
        paper_ids = ["paper1"]
        result = vector_store.retrieve_relevant_chunks(
            query="test query", paper_ids=paper_ids
        )

        # Verify that an empty list is returned since the vector store is not built.
        self.assertEqual(result, [])

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.Vectorstore")
    def test_question_and_answer_success(self, mock_vectorstore, mock_generate_answer):
        """test the main functionality of the question_and_answer tool."""
        # Create a dummy document to simulate a retrieved chunk
        dummy_doc = Document(
            page_content="Dummy content",
            metadata={"paper_id": "paper1", "title": "Paper One", "page": 1},
        )

        # Configure generate_answer to return a dummy answer result
        mock_generate_answer.return_value = {
            "output_text": "Test Answer",
            "papers_used": ["paper1"],
        }

        # Create a dummy embedding model
        dummy_embedding_model = MagicMock(spec=Embeddings)

        # Create a dummy vector store and simulate that it is already built and has the paper loaded
        dummy_vector_store = Vectorstore(embedding_model=dummy_embedding_model)
        dummy_vector_store.vector_store = (
            True  # Simulate that the vector store is built
        )
        dummy_vector_store.loaded_papers.add("paper1")
        dummy_vector_store.retrieve_relevant_chunks = MagicMock(
            return_value=[dummy_doc]
        )
        # Return our dummy vector store when Vectorstore() is instantiated
        mock_vectorstore.return_value = dummy_vector_store

        # Create a dummy LLM model
        dummy_llm_model = MagicMock()

        # Construct the state with required keys
        state = {
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/paper1.pdf",
                    "Title": "Paper One",
                }
            },
            "text_embedding_model": dummy_embedding_model,
            "llm_model": dummy_llm_model,
            "vector_store": dummy_vector_store,
        }

        input_data = {
            "question": "What is the content?",
            "paper_ids": ["paper1"],
            "use_all_papers": False,
            "tool_call_id": "test_tool_call",
            "state": state,
        }
        result = question_and_answer.run(input_data)

        # Verify that generate_answer was called with expected arguments
        mock_generate_answer.assert_called_once()
        args, _ = mock_generate_answer.call_args
        self.assertEqual(args[0], "What is the content?")
        self.assertEqual(args[2], dummy_llm_model)

        # Verify the final response content and tool_call_id in the returned Command
        response_message = result.update["messages"][0]
        expected_output = "Test Answer\n\nSources:\n- Paper One"
        self.assertEqual(response_message.content, expected_output)
        self.assertEqual(response_message.tool_call_id, "test_tool_call")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.Vectorstore")
    def test_question_and_answer_semantic_branch(
        self, mock_vectorstore, mock_generate_answer
    ):
        """test the semantic ranking branch of the question_and_answer tool."""
        # Create a dummy document to simulate a retrieved chunk from semantic ranking
        dummy_doc = Document(
            page_content="Semantic chunk",
            metadata={"paper_id": "paper_sem", "title": "Paper Semantic", "page": 2},
        )

        # Configure generate_answer to return a dummy answer result
        mock_generate_answer.return_value = {
            "output_text": "Semantic Answer",
            "papers_used": ["paper_sem"],
        }

        # Create a dummy Vectorstore instance to simulate the semantic branch behavior
        dummy_vs = MagicMock()
        # Initially, no papers are loaded
        dummy_vs.loaded_papers = set()
        # Explicitly set vector_store to None so that the build_vector_store branch is taken
        dummy_vs.vector_store = None
        # When build_vector_store is called, simulate that the vector store is built
        dummy_vs.build_vector_store.side_effect = lambda: setattr(
            dummy_vs, "vector_store", True
        )
        # Simulate ranking: return a single paper id with score as a tuple for unpacking
        dummy_vs.rank_papers_by_query.return_value = [("paper_sem", 1.0)]
        # Simulate retrieval: return our dummy document
        dummy_vs.retrieve_relevant_chunks.return_value = [dummy_doc]
        # Ensure add_paper is available (it may be called more than once)
        dummy_vs.add_paper.return_value = None

        # When the tool instantiates Vectorstore, return our dummy instance
        mock_vectorstore.return_value = dummy_vs

        # Create dummy embedding and LLM models
        dummy_embedding_model = MagicMock(spec=Embeddings)
        dummy_llm_model = MagicMock()

        # Construct the state WITHOUT a vector_store to force creation,
        # and without explicit paper_ids so the semantic branch is taken.
        state = {
            "article_data": {
                "paper_sem": {
                    "pdf_url": "http://example.com/paper_sem.pdf",
                    "Title": "Paper Semantic",
                }
            },
            "text_embedding_model": dummy_embedding_model,
            "llm_model": dummy_llm_model,
            # Note: "vector_store" key is omitted intentionally
        }

        input_data = {
            "question": "What is semantic content?",
            "paper_ids": None,
            "use_all_papers": False,
            "tool_call_id": "test_semantic_tool_call",
            "state": state,
        }
        result = question_and_answer.run(input_data)

        # Instead of checking that 'vector_store' was added to the original state dict,
        # verify that a new vector store was created by checking that Vectorstore was instantiated.
        mock_vectorstore.assert_called_once_with(embedding_model=dummy_embedding_model)

        # Check that add_paper was called at least once (semantic branch should load the paper)
        self.assertTrue(dummy_vs.add_paper.call_count >= 1)

        # Verify that build_vector_store was called to set up the store
        dummy_vs.build_vector_store.assert_called()

        # Verify that rank_papers_by_query was called with the expected question and top_k=3
        dummy_vs.rank_papers_by_query.assert_called_with(
            "What is semantic content?", top_k=40
        )

        # Verify that retrieve_relevant_chunks was called with the selected paper id.
        dummy_vs.retrieve_relevant_chunks.assert_called_with(
            query="What is semantic content?", paper_ids=["paper_sem"], top_k=25
        )

        # Verify that generate_answer was called with the expected arguments
        mock_generate_answer.assert_called_once()
        args, _ = mock_generate_answer.call_args
        self.assertEqual(args[0], "What is semantic content?")
        self.assertEqual(args[2], dummy_llm_model)

        # Verify that the final response message is correctly
        # formatted with answer and source attribution
        response_message = result.update["messages"][0]
        expected_output = "Semantic Answer\n\nSources:\n- Paper Semantic"
        self.assertEqual(response_message.content, expected_output)
        self.assertEqual(response_message.tool_call_id, "test_semantic_tool_call")

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.Vectorstore")
    def test_question_and_answer_fallback_no_relevant_chunks(self, mock_vectorstore):
        """Test the fallback branch of the question_and_answer
        tool when no relevant chunks are found."""
        # Create a dummy Vectorstore instance to simulate fallback and error conditions.
        dummy_vs = MagicMock()
        # Ensure no papers are loaded initially.
        dummy_vs.loaded_papers = set()
        # Simulate that the vector store is not built.
        dummy_vs.vector_store = None
        # Simulate ranking returning an empty list to force the fallback branch.
        dummy_vs.rank_papers_by_query.return_value = []
        # In the "load selected papers" loop, simulate that add_paper raises an exception.
        dummy_vs.add_paper.side_effect = IOError("Test error")
        # When build_vector_store is called, simulate setting the vector store.
        dummy_vs.build_vector_store.side_effect = lambda: setattr(
            dummy_vs, "vector_store", True
        )
        # Simulate retrieval returning an empty list so that a RuntimeError is raised.
        dummy_vs.retrieve_relevant_chunks.return_value = []
        mock_vectorstore.return_value = dummy_vs

        # Create dummy embedding and LLM models.
        dummy_embedding_model = MagicMock(spec=Embeddings)
        dummy_llm_model = MagicMock()

        # Construct state with article_data containing one paper.
        state = {
            "article_data": {
                "paper1": {
                    "pdf_url": "http://example.com/paper1.pdf",
                    "Title": "Paper One",
                }
            },
            "text_embedding_model": dummy_embedding_model,
            "llm_model": dummy_llm_model,
            # "vector_store" key is omitted intentionally to force creation.
        }

        input_data = {
            "question": "What is fallback test?",
            # Provide paper_ids that do not match article_data so that the
            # fallback branch is triggered.
            "paper_ids": ["nonexistent"],
            "use_all_papers": False,
            "tool_call_id": "test_fallback_call",
            "state": state,
        }

        with self.assertRaises(RuntimeError) as context:
            question_and_answer.run(input_data)

        # Verify that build_vector_store was called to ensure the store is built.
        dummy_vs.build_vector_store.assert_called()

        # Verify that the RuntimeError contains the expected error message.
        self.assertIn(
            "I couldn't find relevant information to answer your question",
            str(context.exception),
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.generate_answer"
    )
    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.Vectorstore")
    def test_question_and_answer_use_all_papers(
        self, mock_vectorstore, mock_generate_answer
    ):
        """test the use_all_papers branch of the question_and_answer tool."""
        # Test the branch where use_all_papers is True.
        # Create a dummy document for retrieval.
        dummy_doc = Document(
            page_content="Content from all papers branch",
            metadata={"paper_id": "paper_all", "title": "Paper All", "page": 1},
        )
        # Configure generate_answer to return a dummy answer.
        mock_generate_answer.return_value = {
            "output_text": "Answer from all papers",
            "papers_used": ["paper_all"],
        }

        # Create a dummy vector store that is already built and already loaded with the paper.
        dummy_vs = MagicMock()
        dummy_vs.vector_store = True
        # Simulate that the paper is already loaded.
        dummy_vs.loaded_papers = {"paper_all"}
        # Simulate retrieval returning the dummy document.
        dummy_vs.retrieve_relevant_chunks.return_value = [dummy_doc]
        # No add_paper call should be needed.
        dummy_vs.add_paper.return_value = None
        # Return our dummy vector store when Vectorstore() is instantiated
        mock_vectorstore.return_value = dummy_vs

        # Construct state with article_data containing one paper and an existing vector_store.
        dummy_embedding_model = MagicMock(spec=Embeddings)
        dummy_llm_model = MagicMock()
        state = {
            "article_data": {
                "paper_all": {
                    "pdf_url": "http://example.com/paper_all.pdf",
                    "Title": "Paper All",
                }
            },
            "text_embedding_model": dummy_embedding_model,
            "llm_model": dummy_llm_model,
            "vector_store": dummy_vs,  # Existing vector store
        }

        input_data = {
            "question": "What is the content from all papers?",
            "paper_ids": None,
            "use_all_papers": True,
            "tool_call_id": "test_use_all_papers",
            "state": state,
        }
        result = question_and_answer.run(input_data)

        # Verify that the use_all_papers branch was
        # taken by checking that all article keys were selected.
        # (This is logged; here we indirectly verify
        # that generate_answer was called with the dummy_llm_model.)
        mock_generate_answer.assert_called_once()
        args, _ = mock_generate_answer.call_args
        self.assertEqual(args[0], "What is the content from all papers?")
        self.assertEqual(args[2], dummy_llm_model)

        # Verify that the final response message includes the answer and source attribution.
        response_message = result.update["messages"][0]
        expected_output = "Answer from all papers\n\nSources:\n- Paper All"
        self.assertEqual(response_message.content, expected_output)
        self.assertEqual(response_message.tool_call_id, "test_use_all_papers")

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.Vectorstore")
    def test_question_and_answer_add_paper_exception(self, mock_vectorstore):
        """test exception handling when add_paper fails."""
        # Test that in the semantic ranking branch, if add_paper raises an exception,
        # the error is logged and then re-raised.
        dummy_vs = MagicMock()
        # No papers are loaded.
        dummy_vs.loaded_papers = set()
        # Simulate that the vector store is not built.
        dummy_vs.vector_store = None
        # In the semantic branch, when trying to load the paper, add_paper will raise an exception.
        dummy_vs.add_paper.side_effect = IOError("Add paper failure")
        # Simulate that build_vector_store would set the store
        # (if reached, but it won't in this test).
        dummy_vs.build_vector_store.side_effect = lambda: setattr(
            dummy_vs, "vector_store", True
        )
        # Ensure retrieval is never reached because add_paper fails.
        dummy_vs.retrieve_relevant_chunks.return_value = []
        mock_vectorstore.return_value = dummy_vs

        dummy_embedding_model = MagicMock(spec=Embeddings)
        dummy_llm_model = MagicMock()
        # Construct state with article_data containing one paper.
        state = {
            "article_data": {
                "paper_err": {
                    "pdf_url": "http://example.com/paper_err.pdf",
                    "Title": "Paper Error",
                }
            },
            "text_embedding_model": dummy_embedding_model,
            "llm_model": dummy_llm_model,
            # No vector_store key provided to force creation of a new one.
        }

        # Use paper_ids=None and use_all_papers=False to trigger semantic ranking branch.
        input_data = {
            "question": "What happens when add_paper fails?",
            "paper_ids": None,
            "use_all_papers": False,
            "tool_call_id": "test_add_paper_exception",
            "state": state,
        }
        with self.assertRaises(IOError) as context:
            question_and_answer.run(input_data)
        self.assertIn("Add paper failure", str(context.exception))

    @patch("aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.PyPDFLoader")
    def test_additional_metadata_field_added(self, mock_pypdf_loader):
        """test that additional metadata fields are added correctly."""
        # Setup the PDF loader to return a single document with empty metadata
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={})
        ]

        # Create a dummy embedding model
        dummy_embedding_model = MagicMock(spec=Embeddings)

        # Define custom metadata fields including an additional field "custom_field"
        custom_fields = ["title", "paper_id", "page", "chunk_id", "custom_field"]
        vector_store = Vectorstore(
            embedding_model=dummy_embedding_model, metadata_fields=custom_fields
        )

        # Paper metadata includes "Title" (for default title) and the additional "custom_field"
        paper_metadata = {"Title": "Test Paper", "custom_field": "custom_value"}

        # Call add_paper to process the document and add metadata
        vector_store.add_paper(
            paper_id="test_paper",
            pdf_url="http://example.com/test.pdf",
            paper_metadata=paper_metadata,
        )

        # Verify that the document was added with the custom field included in its metadata
        self.assertIn("test_paper_0", vector_store.documents)
        added_doc = vector_store.documents["test_paper_0"]
        self.assertEqual(added_doc.metadata.get("custom_field"), "custom_value")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.load_hydra_config"
    )
    def test_generate_answer_missing_config_fields(self, mock_load_config):
        """test that generate_answer raises ValueError for missing config fields."""
        # Create a dummy document and dummy LLM model
        dummy_doc = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )
        dummy_llm_model = MagicMock()

        # Case 1: Configuration is None, expect a ValueError
        mock_load_config.return_value = None
        with self.assertRaises(ValueError) as context_none:
            generate_answer("What is the test?", [dummy_doc], dummy_llm_model)
        self.assertEqual(
            str(context_none.exception), "Hydra config loading failed: config is None."
        )

        # Case 2: Configuration missing 'prompt_template', expect a ValueError
        mock_load_config.return_value = {}
        with self.assertRaises(ValueError) as context_missing:
            generate_answer("What is the test?", [dummy_doc], dummy_llm_model)
        self.assertEqual(
            str(context_missing.exception),
            "The prompt_template is missing from the configuration.",
        )


class TestMissingState(unittest.TestCase):
    """Test error when missing from state."""

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
            question_and_answer.run(tool_input)
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
            question_and_answer.run(tool_input)
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
            question_and_answer.run(tool_input)
        self.assertEqual(str(context.exception), "No article_data found in state.")

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
            question_and_answer.run(tool_input)
        self.assertEqual(str(context.exception), "No article_data found in state.")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.maximal_marginal_relevance"
    )
    def test_retrieve_relevant_chunks_with_filtering(self, mock_mmr):
        """Test that filtering works by paper_ids."""
        mock_mmr.return_value = [0]
        dummy_embedding = [0.1, 0.2, 0.3]

        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = dummy_embedding
        mock_embedding_model.embed_documents.return_value = [dummy_embedding]

        vector_store = Vectorstore(embedding_model=mock_embedding_model)
        vector_store.vector_store = True
        doc1 = Document(page_content="Doc 1", metadata={"paper_id": "paper1"})
        doc2 = Document(page_content="Doc 2", metadata={"paper_id": "paper2"})
        vector_store.documents = {"doc1": doc1, "doc2": doc2}

        results = vector_store.retrieve_relevant_chunks(
            query="query", paper_ids=["paper1"]
        )
        assert len(results) == 1
        assert results[0].metadata["paper_id"] == "paper1"

    def test_retrieve_relevant_chunks_no_matching_docs(self):
        """Ensure it returns empty list and logs warning if no docs match."""
        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.embed_documents.return_value = []

        vector_store = Vectorstore(embedding_model=mock_embedding_model)
        vector_store.vector_store = True
        # Add doc with paper_id that won't match
        vector_store.documents["doc1"] = Document(
            page_content="No match", metadata={"paper_id": "unmatched_paper"}
        )

        results = vector_store.retrieve_relevant_chunks(
            query="test", paper_ids=["nonexistent_id"]
        )
        assert results == []
