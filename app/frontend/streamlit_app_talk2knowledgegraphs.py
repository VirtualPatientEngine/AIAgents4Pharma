#!/usr/bin/env python3

"""
Talk2KnowledgeGraphs: A Streamlit app for the Talk2KnowledgeGraphs graph.
"""

import os
import sys

import hydra
import streamlit as st
from langchain_core.messages import ChatMessage, HumanMessage
from streamlit_feedback import streamlit_feedback
from utils import streamlit_utils

sys.path.append("./")
from aiagents4pharma.talk2knowledgegraphs.agents.t2kg_agent import get_app

# from talk2knowledgegraphs.agents.t2kg_agent import get_app

st.set_page_config(
    page_title="Talk2KnowledgeGraphs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
if "config" not in st.session_state:
    # Load Hydra configuration
    with hydra.initialize(
        version_base=None,
        config_path="../../aiagents4pharma/talk2knowledgegraphs/configs",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["app/frontend=default", "utils/database/milvus=default"],
        )
        st.session_state.config = cfg
else:
    cfg = st.session_state.config


# Resolve logo via shared utility
logo_path = streamlit_utils.resolve_logo(cfg)
if logo_path:
    if hasattr(st, "logo"):
        st.logo(image=logo_path, size="large", link=cfg.app.frontend.logo_link)
    else:
        st.image(image=logo_path, use_column_width=False)

# Provider-aware environment checks will be performed after session init

# Initialize unified session state
streamlit_utils.initialize_session_state(cfg, agent_type="T2KG")

# Provider-aware environment checks (warn-only)
needed_env = set()
llm_choice = st.session_state.get("llm_model", "")

emb_is_openai = False
try:
    from langchain_openai import OpenAIEmbeddings as _OE

    emb_is_openai = isinstance(st.session_state.get("t2kg_emb_model"), _OE)
except Exception:
    emb_is_openai = False


def needs(prefix: str) -> bool:
    return llm_choice.startswith(prefix)


if needs("OpenAI/") or emb_is_openai:
    needed_env.add("OPENAI_API_KEY")
if needs("Azure/"):
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"):
        needed_env.add(var)
if needs("NVIDIA/"):
    needed_env.add("NVIDIA_API_KEY")

missing = [var for var in needed_env if var not in os.environ]
if missing:
    st.warning(
        "Missing environment settings for the selected provider(s): "
        + ", ".join(missing)
    )

# Initialize the app with default LLM model for the first time
if "app" not in st.session_state:
    # Initialize the app
    st.session_state.app = get_app(
        st.session_state.unique_id,
        llm_model=streamlit_utils.get_base_chat_model(st.session_state.llm_model),
    )

# Milvus connection is now handled by backend tools automatically
# No frontend connection management needed

# Get the app
app = st.session_state.app

# Apply custom CSS
streamlit_utils.apply_css()

# Sidebar
with st.sidebar:
    st.markdown("**⚙️ Subgraph Extraction Settings**")
    st.empty()
    # Top-K nodes and edges sliders
    topk_nodes = st.slider(
        "Top-K (Nodes)",
        cfg.app.frontend.reasoning_subgraph_topk_nodes_min,
        cfg.app.frontend.reasoning_subgraph_topk_nodes_max,
        st.session_state.topk_nodes,
        key="st_slider_topk_nodes",
    )
    st.session_state.topk_nodes = topk_nodes
    # Use dedicated edge min/max if present; fall back to node bounds
    edges_min = getattr(
        cfg.app.frontend,
        "reasoning_subgraph_topk_edges_min",
        cfg.app.frontend.reasoning_subgraph_topk_nodes_min,
    )
    edges_max = getattr(
        cfg.app.frontend,
        "reasoning_subgraph_topk_edges_max",
        cfg.app.frontend.reasoning_subgraph_topk_nodes_max,
    )
    topk_edges = st.slider(
        "Top-K (Edges)",
        edges_min,
        edges_max,
        st.session_state.topk_edges,
        key="st_slider_topk_edges",
    )
    st.session_state.topk_edges = topk_edges

# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write(
            """
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2KnowledgeGraphs
            </h3>
            """,
            unsafe_allow_html=True,
        )

        # LLM panel (Only at the front-end for now)
        llms = tuple(streamlit_utils.get_all_available_llms(cfg))
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model,
        )

        # Text embedding model panel (for KG embeddings)
        text_models = tuple(streamlit_utils.get_all_available_embeddings(cfg))
        st.selectbox(
            "Pick a text embedding model",
            text_models,
            index=0,
            key="text_embedding_model",
            on_change=streamlit_utils.update_t2kg_embedding_model,
            help="Used for KG retrieval and related tasks.",
        )

        # Upload files
        streamlit_utils.get_uploaded_files(cfg)

        # Help text
        st.button(
            "Know more ↗",
            on_click=streamlit_utils.help_button,
            use_container_width=False,
        )

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=800):
        st.write("#### 💬 Chat History")

        # Display history of messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(
                    message["content"].role,
                    avatar="🤖" if message["content"].role != "user" else "👩🏻‍💻",
                ):
                    st.markdown(message["content"].content)
                    st.empty()
            elif message["type"] == "button":
                if st.button(message["content"], key=message["key"]):
                    # Trigger the question
                    prompt = message["question"]
                    st.empty()
            elif message["type"] == "plotly":
                streamlit_utils.render_plotly(
                    message["content"],
                    key=message["key"],
                    title=message["title"],
                    #   tool_name=message["tool_name"],
                    save_chart=False,
                )
                st.empty()
            elif message["type"] == "toggle":
                streamlit_utils.render_toggle(
                    key=message["key"],
                    toggle_text=message["content"],
                    toggle_state=message["toggle_state"],
                    save_toggle=False,
                )
                st.empty()
            elif message["type"] == "dataframe":
                streamlit_utils.render_table(
                    message["content"],
                    key=message["key"],
                    # tool_name=message["tool_name"],
                    save_table=False,
                )
                st.empty()
            elif message["type"] == "graph":
                streamlit_utils.render_graph(
                    message["content"], key=message["key"], save_graph=False
                )
                st.empty()

        # Display intro message only the first time
        # i.e. when there are no messages in the chat
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Initializing the agent ..."):
                    config = {"configurable": {"thread_id": st.session_state.unique_id}}

                    # Prepare LLM and embedding model for updating the agent
                    llm_model = streamlit_utils.get_base_chat_model(
                        st.session_state.llm_model
                    )
                    emb_model = st.session_state.get("t2kg_emb_model")

                    # Update the agent state with initial configuration
                    app.update_state(
                        config,
                        {
                            "llm_model": llm_model,
                            "embedding_model": emb_model,
                            "selections": st.session_state.selections,
                            "uploaded_files": st.session_state.uploaded_files,
                            "topk_nodes": st.session_state.topk_nodes,
                            "topk_edges": st.session_state.topk_edges,
                            "dic_source_graph": [
                                {
                                    "name": cfg.utils.database.milvus.milvus_db.database_name,
                                }
                            ],
                        },
                    )

                    intro_prompt = "Tell your name and about yourself. Always start with a greeting."
                    intro_prompt += " and tell about the tools you can run to perform analysis with short description."
                    intro_prompt += " We have provided starter questions (separately) outside your response."
                    intro_prompt += " Do not provide any questions by yourself. Let the users know that they can"
                    intro_prompt += " simply click on the questions to execute them."

                    response = app.stream(
                        {"messages": [HumanMessage(content=intro_prompt)]},
                        config=config,
                        stream_mode="messages",
                    )
                    st.write_stream(streamlit_utils.stream_response(response))
                    current_state = app.get_state(config)
                    # Add response to chat history
                    assistant_msg = ChatMessage(
                        current_state.values["messages"][-1].content, role="assistant"
                    )
                    st.session_state.messages.append(
                        {"type": "message", "content": assistant_msg}
                    )
                    st.empty()

        # Display sample questions when there are few messages
        if len(st.session_state.messages) <= 1:
            for count, question in enumerate(streamlit_utils.sample_questions_t2kg()):
                if st.button(
                    f"Q{count+1}. {question}", key=f"sample_question_{count+1}"
                ):
                    # Trigger the question
                    prompt = question
                # Add button click to chat history
                st.session_state.messages.append(
                    {
                        "type": "button",
                        "question": question,
                        "content": f"Q{count+1}. {question}",
                        "key": f"sample_question_{count+1}",
                    }
                )

        # When the user asks a question
        if prompt:
            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append({"type": "message", "content": prompt_msg})
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

            # Auxiliary visualization-related variables
            graphs_visuals = []
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner():
                    streamlit_utils.get_response(
                        "T2KG", graphs_visuals, app, st, prompt
                    )

            # Visualize the graphs
            if len(graphs_visuals) > 0:
                for count, graph in enumerate(graphs_visuals):
                    streamlit_utils.render_graph(
                        graph_dict=graph["content"], key=graph["key"], save_graph=True
                    )
                    st.empty()

        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=streamlit_utils.submit_feedback,
                key=f"feedback_{st.session_state.run_id}",
            )
            st.empty()
