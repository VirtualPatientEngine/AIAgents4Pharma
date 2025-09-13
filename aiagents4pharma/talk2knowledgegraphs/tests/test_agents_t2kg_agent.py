"""
Test cases for agents/t2kg_agent.py
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.types import Command

from ..agents.t2kg_agent import get_app
from ..tools.milvus_multimodal_subgraph_extraction import (
    MultimodalSubgraphExtractionTool,
)

DATA_PATH = "aiagents4pharma/talk2knowledgegraphs/tests/files"


@pytest.fixture(name="input_dict")
def input_dict_fixture():
    """
    Input dictionary fixture.
    """
    input_dict = {
        "llm_model": None,
        "embedding_model": None,
        "selections": {
            "gene/protein": [],
            "molecular_function": [],
            "cellular_component": [],
            "biological_process": [],
            "drug": [],
            "disease": [],
        },
        "uploaded_files": [
            {
                "file_name": "adalimumab.pdf",
                "file_path": f"{DATA_PATH}/adalimumab.pdf",
                "file_type": "drug_data",
                "uploaded_by": "VPEUser",
                "uploaded_timestamp": "2024-11-05 00:00:00",
            },
        ],
        "topk_nodes": 3,
        "topk_edges": 3,
        "dic_source_graph": [
            {
                "name": "BioBridge",
                "kg_pyg_path": f"{DATA_PATH}/biobridge_multimodal_pyg_graph.pkl",
                "kg_text_path": f"{DATA_PATH}/biobridge_multimodal_text_graph.pkl",
            }
        ],
        "dic_extracted_graph": [],
    }
    return input_dict


def mock_milvus_collection(name):
    """
    Mock Milvus collection for testing.
    """
    # name is intentionally unused in this simplified mock
    del name
    nodes = MagicMock()
    nodes.query.return_value = [
        {
            "node_index": 0,
            "node_id": "id1",
            "node_name": "Adalimumab",
            "node_type": "drug",
            "feat": "featA",
            "feat_emb": [0.1, 0.2, 0.3],
            "desc": "descA",
            "desc_emb": [0.1, 0.2, 0.3],
        },
        {
            "node_index": 1,
            "node_id": "id2",
            "node_name": "TNF",
            "node_type": "gene/protein",
            "feat": "featB",
            "feat_emb": [0.4, 0.5, 0.6],
            "desc": "descB",
            "desc_emb": [0.4, 0.5, 0.6],
        },
    ]
    nodes.load.return_value = None

    edges = MagicMock()
    edges.query.return_value = [
        {
            "triplet_index": 0,
            "head_id": "id1",
            "head_index": 0,
            "tail_id": "id2",
            "tail_index": 1,
            "edge_type": "drug,acts_on,gene/protein",
            "display_relation": "acts_on",
            "feat": "featC",
            "feat_emb": [0.7, 0.8, 0.9],
        }
    ]
    edges.load.return_value = None

    # Default path in tests expects None for unknown collections (implicit)


def _invoke_app_with_mocks(unique_id, input_dict):
    """Run the app with patched Milvus + tool stack and return (app, config, response)."""
    app = get_app(unique_id, llm_model=input_dict["llm_model"])
    config = {"configurable": {"thread_id": unique_id}}
    app.update_state(config, input_dict)
    prompt = (
        "Adalimumab is a fully human monoclonal antibody (IgG1) that "
        "specifically binds to tumor necrosis factor-alpha (TNF-α), a "
        "pro-inflammatory cytokine.\n\n"
        "I would like to get evidence from the knowledge graph about the "
        "mechanism of actions related to Adalimumab in treating inflammatory "
        "bowel disease (IBD). Please follow these steps:\n"
        "- Extract a subgraph from the PrimeKG that contains information about "
        "Adalimumab.\n- Summarize the extracted subgraph.\n"
        "- Reason about the mechanism of action of Adalimumab in treating IBD.\n\n"
        "Please set the extraction name for the extraction process as `subkg_"
        f"{unique_id}`."
    )

    mocks = {
        "pcst": MagicMock(),
        "connections": MagicMock(),
        "compose": MagicMock(),
        "connections_manager": MagicMock(),
        "db": MagicMock(),
        "conn_mgr": MagicMock(),
    }

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.Collection",
                side_effect=mock_milvus_collection,
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.MultimodalPCSTPruning",
                mocks["pcst"],
            )
        )
        stack.enter_context(patch("pymilvus.connections", mocks["connections"]))
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.initialize"
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.hydra.compose",
                mocks["compose"],
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.database."
                "milvus_connection_manager.connections",
                mocks["connections_manager"],
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.database."
                "milvus_connection_manager.Collection",
                side_effect=mock_milvus_collection,
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.utils.database.milvus_connection_manager.db",
                mocks["db"],
            )
        )
        stack.enter_context(
            patch(
                "aiagents4pharma.talk2knowledgegraphs.tools."
                "milvus_multimodal_subgraph_extraction.MilvusConnectionManager",
                mocks["conn_mgr"],
            )
        )

        def mock_tool_execution(tool_call_id, state, prompt, arg_data=None):
            del prompt, arg_data
            mock_extracted_graph = {
                "name": f"subkg_{unique_id}",
                "tool_call_id": tool_call_id,
                "graph_source": "BioBridge",
                "topk_nodes": 3,
                "topk_edges": 3,
                "graph_dict": {
                    "name": "extracted_subgraph",
                    "nodes": ["Adalimumab", "TNF"],
                    "edges": [("Adalimumab", "acts_on", "TNF")],
                },
                "graph_text": (
                    "Adalimumab acts on TNF protein for treating inflammatory diseases."
                ),
                "graph_summary": None,
            }
            tool_message = ToolMessage(
                content=(
                    "Subgraph extraction completed successfully. "
                    "Extracted subgraph containing Adalimumab and TNF interactions."
                ),
                tool_call_id=tool_call_id,
                name="subgraph_extraction",
            )
            return Command(
                update={
                    "messages": [tool_message],
                    "dic_extracted_graph": state.get("dic_extracted_graph", [])
                    + [mock_extracted_graph],
                }
            )

        stack.enter_context(
            patch.object(MultimodalSubgraphExtractionTool, "_run", side_effect=mock_tool_execution)
        )

        # set return values via the mocks dict
        mocks["connections"].has_connection.return_value = True
        mocks["connections_manager"].has_connection.return_value = True
        mocks["db"].using_database.return_value = None

        pcst_instance = MagicMock()
        pcst_instance.extract_subgraph.return_value = {
            "nodes": pd.Series([0, 1]),
            "edges": pd.Series([0]),
        }
        mocks["pcst"].return_value = pcst_instance

        cfg = MagicMock()
        for k, v in {
            "cost_e": 1.0,
            "c_const": 1.0,
            "root": 0,
            "num_clusters": 1,
            "pruning": True,
            "verbosity_level": 0,
            "search_metric_type": "L2",
        }.items():
            setattr(cfg, k, v)
        cfg.node_colors_dict = {"drug": "blue", "gene/protein": "red"}

        mocks["compose"].return_value = MagicMock()
        mocks["compose"].return_value.tools.multimodal_subgraph_extraction = cfg
        mocks[
            "compose"
        ].return_value.tools.subgraph_summarization.prompt_subgraph_summarization = (
            "Summarize the following subgraph: {textualized_subgraph}"
        )

        db_cfg = MagicMock()
        for k, v in {
            "alias": "test_alias",
            "host": "localhost",
            "port": "19530",
            "user": "root",
            "password": "password",
            "database_name": "test_db",
        }.items():
            setattr(db_cfg.milvus_db, k, v)
        mocks["compose"].return_value.utils.database.milvus = db_cfg.milvus_db

        conn = MagicMock()
        conn.ensure_connection.return_value = True
        conn.get_connection_info.return_value = {"database": "test_db", "connected": True}
        conn.test_connection.return_value = True
        mocks["conn_mgr"].return_value = conn

        response = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    return app, config, response


def test_t2kg_agent_openai_milvus_mock(input_dict):
    """
    Test the T2KG agent using OpenAI model and Milvus mock.

    Args:
        input_dict: Input dictionary
    """
    input_dict["llm_model"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    input_dict["embedding_model"] = OpenAIEmbeddings(model="text-embedding-3-small")
    unique_id = 12345
    app, config, response = _invoke_app_with_mocks(unique_id, input_dict)

    assert isinstance(response["messages"][-1].content, str)
    dic_extracted_graph = app.get_state(config).values["dic_extracted_graph"][0]
    assert isinstance(dic_extracted_graph, dict)
    assert dic_extracted_graph["name"] == "subkg_12345"
    assert dic_extracted_graph["graph_source"] == "BioBridge"
    assert dic_extracted_graph["topk_nodes"] == 3
    assert dic_extracted_graph["topk_edges"] == 3
    assert isinstance(dic_extracted_graph["graph_dict"], dict)
    assert len(dic_extracted_graph["graph_dict"]["nodes"]) > 0
    assert len(dic_extracted_graph["graph_dict"]["edges"]) > 0
    assert isinstance(dic_extracted_graph["graph_text"], str)
    assert isinstance(dic_extracted_graph["graph_summary"], str)
    assert "Adalimumab" in response["messages"][-1].content
    assert "TNF" in response["messages"][-1].content

    # Another test for unknown collection
    assert mock_milvus_collection("unknown") is None
