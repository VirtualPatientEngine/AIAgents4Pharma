"""
Test cases for tools/milvus_multimodal_subgraph_extraction.py
"""

import unittest
from unittest.mock import patch, MagicMock
from ..tools.milvus_multimodal_subgraph_extraction import MultimodalSubgraphExtractionTool

DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


class TestMultimodalSubgraphExtractionTool(unittest.TestCase):
    """
    Test the MultimodalSubgraphExtractionTool.
    """
    def setUp(self):
        self.prompt = """
        Extract all relevant information related to nodes of genes related to inflammatory bowel disease
        (IBD) that existed in the knowledge graph.
        Please set the extraction name for this process as `subkg_12345`.
        """
        self.agent_state = {
            "selections": {
                "gene/protein": [],
                "molecular_function": [],
                "cellular_component": [],
                "biological_process": [],
                "drug": [],
                "disease": []
            },
            "uploaded_files": [],
            "topk_nodes": 3,
            "topk_edges": 3,
            "dic_source_graph": [{
                "name": "BioBridge",
                "kg_pyg_path": f"{DATA_PATH}/biobridge_multimodal_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/biobridge_multimodal_text_graph.pkl",
            }],
            "embedding_model": MagicMock(embed_query=lambda prompt: [0.1, 0.2, 0.3])
        }

    def _mock_milvus_query(self, expr=None, output_fields=None):
        if 'node_name' in expr:
            return [{
                "node_id": "P12345",
                "node_name": "GeneX",
                "node_type": "gene/protein",
                "feat": "text1",
                "feat_emb": [0.1, 0.2, 0.3],
                "desc": "desc1",
                "desc_emb": [0.4, 0.5, 0.6]
            }]
        elif 'node_index' in expr:
            return [{
                "node_id": "P12345",
                "node_name": "GeneX",
                "node_type": "gene/protein",
                "desc": "desc1"
            }]
        elif 'triplet_index' in expr:
            return [{
                "head_id": "P12345",
                "tail_id": "D56789",
                "edge_type": "associated_with|related_to"
            }]
        return []

    @patch("aiagents4pharma.talk2knowledgegraphs.tools." + \
        "milvus_multimodal_subgraph_extraction.connections.has_connection", return_value=True)
    @patch("aiagents4pharma.talk2knowledgegraphs.tools." + \
        "milvus_multimodal_subgraph_extraction.Collection")
    def test_extract_multimodal_subgraph_wo_doc(self, mock_coll, mock_has_connection):
        """
        Test the extraction of a multimodal subgraph without document context.
        """
        # Setup Collection mock
        mock_coll = MagicMock()
        mock_coll.query.side_effect = self._mock_milvus_query
        mock_coll.return_value = mock_coll

        # Instantiate and invoke
        tool = MultimodalSubgraphExtractionTool()
        result = tool.invoke({
            "tool_call_id": "subgraph_extraction_tool",
            "state": self.agent_state,
            "prompt": self.prompt,
            "arg_data": {"extraction_name": "subkg_12345"}
        })

        graph = result.update["dic_extracted_graph"][0]
        self.assertEqual(graph["name"], "subkg_12345")
        self.assertGreater(len(graph["graph_dict"]["nodes"]), 0)
        self.assertGreater(len(graph["graph_dict"]["edges"]), 0)

    # @patch("aiagents4pharma.talk2knowledgegraphs.tools.milvus_multimodal_subgraph_extraction.Collection")
    # def test_extract_multimodal_subgraph_w_doc(self, mock_coll):
    #     """
    #     Test the extraction of a multimodal subgraph with document context.
    #     """
    #     # Add mock uploaded file
    #     self.agent_state["uploaded_files"] = [{
    #         "file_name": "multimodal-analysis.xlsx",
    #         "file_path": f"{DATA_PATH}/multimodal-analysis.xlsx",
    #         "file_type": "multimodal",
    #         "uploaded_by": "VPEUser",
    #         "uploaded_timestamp": "2025-05-12 00:00:00",
    #     }]

    #     # Setup Collection mock
    #     mock_coll = MagicMock()
    #     mock_coll.query.side_effect = self._mock_milvus_query
    #     mock_coll.return_value = mock_coll

    #     # Instantiate and invoke
    #     tool = MultimodalSubgraphExtractionTool()
    #     result = tool.invoke({
    #         "tool_call_id": "subgraph_extraction_tool",
    #         "state": self.agent_state,
    #         "prompt": self.prompt,
    #         "arg_data": {"extraction_name": "subkg_12345"}
    #     })

    #     graph = result.update["dic_extracted_graph"][0]
    #     self.assertEqual(graph["name"], "subkg_12345")
    #     self.assertGreater(len(graph["graph_dict"]["nodes"]), 0)
    #     self.assertGreater(len(graph["graph_dict"]["edges"]), 0)
