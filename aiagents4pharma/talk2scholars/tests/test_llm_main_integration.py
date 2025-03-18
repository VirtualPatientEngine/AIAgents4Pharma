"""
Integration tests for talk2scholars system with OpenAI.
This test triggers all sub-agents by sending a conversation that covers:
- Searching Semantic Scholar (S2 agent)
- Retrieving Zotero results (Zotero agent)
- Querying PDF content (PDF agent)
- Downloading paper details from arXiv (Paper Download agent)
"""

# This will be covered in the next pr.

#
# import os
# import pytest
# import hydra
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage
# from ..agents.main_agent import get_app
# from ..state.state_talk2scholars import Talk2Scholars
#
# # pylint: disable=redefined-outer-name,too-few-public-methods
#
#
# @pytest.mark.skipif(
#     not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key to run"
# )
# def test_main_agent_real_llm():
#     """
#     Integration test for the Talk2Scholars system using a real OpenAI LLM.
#     This test verifies that the supervisor correctly routes to all sub-agents by
#     providing a conversation with queries intended to trigger each agent.
#     """
#     # Load Hydra configuration EXACTLY like in main_agent.py
#     with hydra.initialize(version_base=None, config_path="../configs"):
#         cfg = hydra.compose(
#             config_name="config", overrides=["agents/talk2scholars/main_agent=default"]
#         )
#     hydra_cfg = cfg.agents.talk2scholars.main_agent
#     assert hydra_cfg is not None, "Hydra config failed to load"
#
#     # Use the real OpenAI API (ensure OPENAI_API_KEY is set in environment)
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=hydra_cfg.temperature)
#
#     # Initialize the main agent workflow (with real Hydra config)
#     thread_id = "test_thread"
#     app = get_app(thread_id, llm)
#
#     # Provide a multi-turn conversation intended to trigger all sub-agents:
#     # - S2 agent: "Search Semantic Scholar for AI papers on transformers."
#     # - Zotero agent: "Retrieve Zotero results for these papers."
#     # - PDF agent: "Analyze the attached PDF and summarize its key findings."
#     # - Paper Download agent: "Download the paper details from arXiv."
#     initial_state = Talk2Scholars(
#         messages=[
#             HumanMessage(
#                 content="Search Semantic Scholar for AI papers on transformers."
#             ),
#             HumanMessage(content="Also, retrieve Zotero results for these papers."),
#             HumanMessage(
#                 content="I have attached a PDF; analyze it and tell me the key findings."
#             ),
#             HumanMessage(content="Finally, download the paper from arXiv."),
#         ]
#     )
#
#     # Invoke the agent (which routes to the appropriate sub-agents)
#     result = app.invoke(
#         initial_state,
#         {"configurable": {"config_id": thread_id, "thread_id": thread_id}},
#     )
#
#     # Assert that the result contains messages and that the final message is valid.
#     assert "messages" in result, "Expected 'messages' in the response"
#     last_message = result["messages"][-1]
#     assert isinstance(
#         last_message, (HumanMessage, AIMessage, str)
#     ), "Last message should be a valid response type"
#
#     # Concatenate message texts (if available) to perform keyword checks.
#     output_text = " ".join(
#         msg.content if hasattr(msg, "content") else str(msg)
#         for msg in result["messages"]
#     ).lower()
#
#     # Check for keywords that suggest each sub-agent was invoked.
#     for keyword in ["semantic scholar", "zotero", "pdf", "arxiv"]:
#         assert (
#             keyword in output_text
#         ), f"Expected keyword '{keyword}' in the output response"
