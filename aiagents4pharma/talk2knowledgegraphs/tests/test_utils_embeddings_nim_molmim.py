#!/usr/bin/env python3

"""
Test cases for utils/embeddings/nim_molmim.py
"""

import unittest
from unittest.mock import patch, MagicMock
from ..utils.embeddings.nim_molmim import EmbeddingWithMOLMIM

class TestEmbeddingWithMOLMIM(unittest.TestCase):
    """
    Test cases for EmbeddingWithMOLMIM class.
    """
    def setUp(self):
        self.base_url = "https://fake-nim-api.com/embeddings"
        self.embeddings_model = EmbeddingWithMOLMIM(self.base_url)
        self.test_texts = ["CCO", "CCC", "C=O"]
        self.test_query = "CCO"
        self.mock_response = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        }

    @patch("requests.post")
    def test_embed_documents(self, mock_post):
        '''
        Test the embed_documents method.
        '''
        # Mock the response from requests.post
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = self.mock_response
        embeddings = self.embeddings_model.embed_documents(self.test_texts)
        # Assertions
        self.assertEqual(embeddings, self.mock_response["embeddings"])
        mock_post.assert_called_once_with(
            self.base_url,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            data='{"sequences": ["CCO", "CCC", "C=O"]}',
            timeout=60
        )

    @patch("requests.post")
    def test_embed_query(self, mock_post):
        '''
        Test the embed_query method.
        '''
        # Mock the response from requests.post
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        embedding = self.embeddings_model.embed_query(self.test_query)
        # Assertions
        self.assertEqual(embedding, [[0.1, 0.2, 0.3]])
        mock_post.assert_called_once_with(
            self.base_url,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            data='{"sequences": ["CCO"]}',
            timeout=60
        )
