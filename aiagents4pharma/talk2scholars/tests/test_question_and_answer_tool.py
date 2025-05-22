"""
Unit tests for question_and_answer tool functionality.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.question_and_answer import (
    question_and_answer,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.generate_answer import (
    generate_answer,
    load_hydra_config,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker import (
    rank_papers_by_query,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks import (
    retrieve_relevant_chunks,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store import Vectorstore


class TestQuestionAndAnswerTool(unittest.TestCase):
    """tests for question_and_answer tool functionality."""

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.PyPDFLoader")
    def test_add_paper(self, mock_pypdf_loader):
        """test adding a paper to the vector store."""
        # Mock the PDF loader
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [Document(page_content="Page content")]

        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(
            embedding_model=mock_embedding_model,
            config=load_hydra_config(),
        )

        # Add a paper
        vector_store.add_paper(
            paper_id="test_paper",
            pdf_url="http://example.com/test.pdf",
            paper_metadata={"Title": "Test Paper"},
        )

        # Check if the paper was added
        self.assertIn("test_paper_0", vector_store.documents)

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.PyPDFLoader")
    def test_add_paper_already_loaded(self, mock_pypdf_loader):
        """Test that adding a paper that is already loaded does not re-load or add new documents."""
        # Mock the PDF loader (it should not be used when the paper is already loaded)
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [Document(page_content="Page content")]

        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore
        vector_store = Vectorstore(
            embedding_model=mock_embedding_model,
            config=load_hydra_config(),
        )

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
        "aiagents4pharma.talk2scholars.tools.pdf.utils.nvidia_nim_reranker.NVIDIARerank"
    )
    def test_rank_papers_by_query(self, mock_nvidia_rerank):
        """test ranking papers by query."""
        # Create a mock config object with the top_k_papers attribute
        # Create a mock config object with required reranker settings and top_k_papers
        mock_config = SimpleNamespace(
            reranker=SimpleNamespace(model="dummy", api_key="key"),
            top_k_papers=1,
        )

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

        # Rank papers using the standalone function
        ranked_papers = rank_papers_by_query(
            vector_store, "test query", mock_config, top_k=mock_config.top_k_papers
        )

        # Check if the ranking is correct (updated expectation: a list of paper IDs)
        self.assertEqual(ranked_papers[0], "test_paper")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.maximal_marginal_relevance"
    )
    def test_retrieve_relevant_chunks(self, mock_mmr):
        """Test retrieving relevant chunks without filters."""
        mock_mmr.return_value = [0]
        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        vector_store = Vectorstore(embedding_model=mock_embedding_model)
        vector_store.vector_store = True
        # Add a document chunk with required metadata including chunk_id
        vector_store.documents["test_doc"] = Document(
            page_content="Test content",
            metadata={"paper_id": "test_paper", "chunk_id": 0},
        )

        results = retrieve_relevant_chunks(vector_store, query="test query")
        assert len(results) == 1
        assert results[0].metadata["paper_id"] == "test_paper"

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.utils.generate_answer.BaseChatModel"
    )
    def test_generate_answer(self, mock_base_chat_model):
        """test generating an answer."""
        # Mock the language model
        mock_llm = mock_base_chat_model.return_value
        mock_llm.invoke.return_value.content = "Generated answer"

        # Create a mock document
        mock_document = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )

        # Generate answer with dummy config
        config = {"prompt_template": "{context} {question}"}
        result = generate_answer(
            question="What is the test?",
            retrieved_chunks=[mock_document],
            llm_model=mock_llm,
            config=config,
        )

        # Check if the answer is generated correctly
        self.assertEqual(result["output_text"], "Generated answer")

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.PyPDFLoader")
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

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.PyPDFLoader")
    def test_add_paper_missing_config(self, mock_pypdf_loader):
        """Test that add_paper raises ValueError when config is missing."""
        # Mock the PDF loader to return a single page
        mock_loader = mock_pypdf_loader.return_value
        mock_loader.load.return_value = [Document(page_content="Page content")]

        # Mock embedding model
        mock_embedding_model = MagicMock(spec=Embeddings)

        # Initialize Vectorstore without config (default None)
        vector_store = Vectorstore(embedding_model=mock_embedding_model)

        # Attempt to add a paper and expect a configuration error
        with self.assertRaises(ValueError) as cm:
            vector_store.add_paper(
                paper_id="test_paper",
                pdf_url="http://example.com/test.pdf",
                paper_metadata={"Title": "Test Paper"},
            )
        self.assertEqual(
            str(cm.exception),
            "Configuration is required for text splitting in Vectorstore.",
        )

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
        result = retrieve_relevant_chunks(vector_store, query="test query")

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
        # Use module-level retrieve_relevant_chunks

        result = retrieve_relevant_chunks(
            vector_store, query="test query", paper_ids=paper_ids
        )

        # Verify that an empty list is returned since the vector store is not built.
        self.assertEqual(result, [])

    @patch("aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store.PyPDFLoader")
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
            embedding_model=dummy_embedding_model,
            metadata_fields=custom_fields,
            config=load_hydra_config(),
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

    def test_generate_answer_missing_config_fields(self):
        """test that generate_answer raises ValueError for missing config fields."""
        # Create a dummy document and dummy LLM model
        dummy_doc = Document(
            page_content="Test content", metadata={"paper_id": "test_paper"}
        )
        dummy_llm_model = MagicMock()

        # Case 1: Configuration is None, expect a ValueError
        with self.assertRaises(ValueError) as context_none:
            generate_answer("What is the test?", [dummy_doc], dummy_llm_model, None)
        self.assertEqual(
            str(context_none.exception),
            "Configuration for generate_answer is required.",
        )

        # Case 2: Configuration missing 'prompt_template', expect a ValueError
        with self.assertRaises(ValueError) as context_missing:
            generate_answer("What is the test?", [dummy_doc], dummy_llm_model, {})
        self.assertEqual(
            str(context_missing.exception),
            "The prompt_template is missing from the configuration.",
        )

    def test_state_validation_errors(self):
        """Test errors raised for missing state entries."""
        valid_articles = {"paper1": {"pdf_url": "u", "Title": "T1"}}
        cases = [
            ({"llm_model": MagicMock(), "article_data": valid_articles},
             "No text embedding model found in state."),
            ({"text_embedding_model": MagicMock(), "article_data": valid_articles},
             "No LLM model found in state."),
            ({"text_embedding_model": MagicMock(), "llm_model": MagicMock()},
             "No article_data found in state."),
            ({"text_embedding_model": MagicMock(), "llm_model": MagicMock(), "article_data": {}},
             "No article_data found in state."),
        ]
        for state_dict, expected_msg in cases:
            with self.subTest(state=state_dict):
                tool_input = {"question": "Q?", "state": state_dict, "tool_call_id": "id"}
                with self.assertRaises(ValueError) as cm:
                    question_and_answer.run(tool_input)
                self.assertEqual(str(cm.exception), expected_msg)

    def test_retrieve_relevant_chunks_with_filtering(self):
        """Test that filtering works by paper_ids."""
        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        vector_store = Vectorstore(
            embedding_model=mock_embedding_model, config=load_hydra_config()
        )
        vector_store.vector_store = True
        # Add document chunks with necessary metadata including chunk_ids
        doc1 = Document(
            page_content="Doc 1", metadata={"paper_id": "paper1", "chunk_id": 0}
        )
        doc2 = Document(
            page_content="Doc 2", metadata={"paper_id": "paper2", "chunk_id": 1}
        )
        vector_store.documents = {"doc1": doc1, "doc2": doc2}

        results = retrieve_relevant_chunks(
            vector_store, query="query", paper_ids=["paper1"]
        )
        assert len(results) == 1
        assert results[0].metadata["paper_id"] == "paper1"

    def test_retrieve_relevant_chunks_no_matching_docs(self):
        """Ensure it returns empty list and logs warning if no docs match."""
        mock_embedding_model = MagicMock(spec=Embeddings)
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.embed_documents.return_value = []

        vector_store = Vectorstore(
            embedding_model=mock_embedding_model, config=load_hydra_config()
        )
        vector_store.vector_store = True
        # Add doc with paper_id that won't match
        vector_store.documents["doc1"] = Document(
            page_content="No match", metadata={"paper_id": "unmatched_paper"}
        )
        # Use util function for retrieval
        results = retrieve_relevant_chunks(
            vector_store, query="test", paper_ids=["nonexistent_id"]
        )
        assert results == []

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer."
        "helper.get_state_models_and_data"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer."
        "helper.init_vector_store"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer."
        "retrieve_relevant_chunks"
    )
    @patch.multiple(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.helper",
        run_reranker=lambda vs, query, candidates: ["p1"],
        format_answer=lambda question, chunks, llm, articles: "formatted answer",
    )
    def test_question_and_answer_happy_path(
        self, mock_retrieve, mock_init, mock_state
    ):
        """Test happy path for question_and_answer tool."""
        # Setup helper and utility mocks
        emb = object()
        llm = object()
        articles = {"p1": {"pdf_url": "u"}}
        mock_state.return_value = (emb, llm, articles)
        # Provide dummy vector store for loading
        vs = SimpleNamespace(loaded_papers=set(), add_paper=MagicMock())
        mock_init.return_value = vs
        # Dummy chunk list for retrieval
        dummy_chunk = Document(page_content="c", metadata={"paper_id": "p1"})
        mock_retrieve.return_value = [dummy_chunk]

        # Use module-level question_and_answer

        state = {}
        tool_input = {"question": "Q?", "state": state, "tool_call_id": "tid"}
        result = question_and_answer.run(tool_input)
        # Verify Command message content and tool_call_id
        msgs = result.update.get("messages", [])
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertEqual(msg.content, "formatted answer")
        self.assertEqual(msg.tool_call_id, "tid")

    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.helper."
        "get_state_models_and_data"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.helper.init_vector_store"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.helper.run_reranker",
        return_value=["p1"],
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.pdf.question_and_answer.retrieve_relevant_chunks",
        return_value=[],
    )
    def test_question_and_answer_no_chunks(
        self, _mock_retrieve, _mock_rerank, mock_init, mock_state
    ):
        """Test that no chunks raises RuntimeError."""
        emb = object()
        llm = object()
        articles = {"p1": {"pdf_url": "u"}}
        mock_state.return_value = (emb, llm, articles)
        # Provide dummy vector store to satisfy load_candidate_papers
        vs = SimpleNamespace(loaded_papers=set(), add_paper=MagicMock())
        mock_init.return_value = vs

        tool_input = {"question": "Q?", "state": {}, "tool_call_id": "id"}
        with self.assertRaises(RuntimeError) as cm:
            question_and_answer.run(tool_input)
        self.assertIn("No relevant chunks found for question", str(cm.exception))
