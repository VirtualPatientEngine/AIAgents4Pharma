#!/usr/bin/env python3

"""
Talk2KnowledgeGraphs: A Streamlit app for the Talk2KnowledgeGraphs graph.
"""

import os
import sys
import random
import streamlit as st
import pandas as pd
import hydra
from streamlit_feedback import streamlit_feedback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.tracers.context import collect_runs
from langchain.callbacks.tracers import LangChainTracer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
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
        cfg = hydra.compose(config_name="config", overrides=["app/frontend=default"])
        cfg = cfg.app.frontend
        st.session_state.config = cfg
else:
    cfg = st.session_state.config


# st.logo(
#     image='docs/VPE.png',
#     size='large',
#     link='https://github.com/VirtualPatientEngine'
# )

# Check if env variable OPENAI_API_KEY exists
if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "Please set the OPENAI_API_KEY environment \
        variable in the terminal where you run the app."
    )
    st.stop()

# Initialize current user
if "current_user" not in st.session_state:
    st.session_state.current_user = cfg.default_user

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for SBML file uploader
# if "sbml_key" not in st.session_state:
#     st.session_state.sbml_key = 0

# Initialize session state for selections
if "selections" not in st.session_state:
    st.session_state.selections = streamlit_utils.initialize_selections()

# Initialize session state for pre-clinical data package uploader
if "data_package_key" not in st.session_state:
    st.session_state.data_package_key = 0

# Initialize session state for multimodal data package uploader
if "multimodal_key" not in st.session_state:
    st.session_state.multimodal_key = 0

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

    # Make directories if not exists
    os.makedirs(cfg.upload_data_dir, exist_ok=True)

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = "T2KG-" + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)

# Initialize the LLM model
if "llm_model" not in st.session_state:
    st.session_state.llm_model = tuple(cfg.openai_llms + cfg.ollama_llms)[0]

# Initialize the app with default LLM model for the first time
if "app" not in st.session_state:
    # Initialize the app
    if st.session_state.llm_model in cfg.openai_llms:
        print("Using OpenAI LLM model")
        st.session_state.app = get_app(
            st.session_state.unique_id,
            llm_model=ChatOpenAI(
                model=st.session_state.llm_model, temperature=cfg.temperature
            ),
        )
    else:
        print("Using Ollama LLM model")
        st.session_state.app = get_app(
            st.session_state.unique_id,
            llm_model=ChatOllama(
                model=st.session_state.llm_model, temperature=cfg.temperature
            ),
        )

if "topk_nodes" not in st.session_state:
    # Subgraph extraction settings
    st.session_state.topk_nodes = cfg.reasoning_subgraph_topk_nodes
    st.session_state.topk_edges = cfg.reasoning_subgraph_topk_edges

if "milvus_connection" not in st.session_state:
    st.session_state.milvus_connection = streamlit_utils.setup_milvus(cfg)
    print("Milvus connection established:", st.session_state.milvus_connection)

# Get the app
app = st.session_state.app

# Apply custom CSS
streamlit_utils.apply_css()

# Sidebar
with st.sidebar:
    st.markdown("**⚙️ Subgraph Extraction Settings**")
    topk_nodes = st.slider(
        "Top-K (Nodes)",
        cfg.reasoning_subgraph_topk_nodes_min,
        cfg.reasoning_subgraph_topk_nodes_max,
        st.session_state.topk_nodes,
        key="st_slider_topk_nodes",
    )
    st.session_state.topk_nodes = topk_nodes
    topk_edges = st.slider(
        "Top-K (Edges)",
        cfg.reasoning_subgraph_topk_nodes_min,
        cfg.reasoning_subgraph_topk_nodes_max,
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
        # llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        llms = tuple(cfg.openai_llms + cfg.ollama_llms)
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model,
        )

        # Upload files
        streamlit_utils.get_uploaded_files(cfg)

        # Help text
        # st.button("Know more ↗",
        #         #   icon="ℹ️",
        #           on_click=streamlit_utils.help_button,
        #           use_container_width=False)

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=800):
        st.write("#### 💬 Chat History")

        # Display chat messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(
                    message["content"].role,
                    avatar="🤖" if message["content"].role != "user" else "👩🏻‍💻",
                ):
                    st.markdown(message["content"].content)
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
                # with st.spinner("Fetching response ..."):
                with st.spinner():
                    # Get chat history
                    history = [
                        (m["content"].role, m["content"].content)
                        for m in st.session_state.messages
                        if m["type"] == "message"
                    ]
                    # Convert chat history to ChatMessage objects
                    chat_history = [
                        SystemMessage(content=m[1])
                        if m[0] == "system"
                        else HumanMessage(content=m[1])
                        if m[0] == "human"
                        else AIMessage(content=m[1])
                        for m in history
                    ]

                    # Prepare LLM and embedding model for updating the agent
                    if st.session_state.llm_model in cfg.openai_llms:
                        llm_model = ChatOpenAI(
                            model=st.session_state.llm_model,
                            temperature=cfg.temperature,
                        )
                        # emb_model = OpenAIEmbeddings(model=cfg.openai_embeddings[0])
                        # For IBD BioBridge data, we still use Ollama embeddings
                        emb_model = OllamaEmbeddings(model=cfg.ollama_embeddings[0])
                    else:
                        llm_model = ChatOllama(
                            model=st.session_state.llm_model,
                            temperature=cfg.temperature,
                        )
                        emb_model = OllamaEmbeddings(model=cfg.ollama_embeddings[0])

                    # Create config for the agent
                    config = {"configurable": {"thread_id": st.session_state.unique_id}}
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
                                    "name": st.session_state.config["kg_name"],
                                    "kg_pyg_path": st.session_state.config["kg_pyg_path"],
                                    "kg_text_path": st.session_state.config["kg_text_path"],
                                }
                            ],
                        },
                    )

                    # Update the agent states
                    current_state = app.get_state(config)

                    ERROR_FLAG = False
                    with collect_runs() as cb:
                        # Add Langsmith tracer
                        tracer = LangChainTracer(
                            project_name=st.session_state.project_name
                        )
                        # Get response from the agent
                        response = app.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config | {"callbacks": [tracer]},
                        )
                        st.session_state.run_id = cb.traced_runs[-1].id
                    current_state = app.get_state(config)

                    # Add response to chat history
                    assistant_msg = ChatMessage(
                        response["messages"][-1].content, role="assistant"
                    )
                    st.session_state.messages.append(
                        {"type": "message", "content": assistant_msg}
                    )
                    # Display the response in the chat
                    st.markdown(response["messages"][-1].content)
                    st.empty()

                    # Get the current state of the graph
                    current_state = app.get_state(config)

                    # # Get the messages from the current state
                    # # and reverse the order
                    reversed_messages = current_state.values["messages"][::-1]

                    # Loop through the reversed messages until a
                    # HumanMessage is found i.e. the last message
                    # from the user. This is to display the results
                    # of the tool calls made by the agent since the
                    # last message from the user.
                    for msg in reversed_messages:
                        # print (msg)
                        # Break the loop if the message is a HumanMessage
                        # i.e. the last message from the user
                        if isinstance(msg, HumanMessage):
                            break
                        # Skip the message if it is an AIMessage
                        # i.e. a message from the agent. An agent
                        # may make multiple tool calls before the
                        # final response to the user.
                        if isinstance(msg, AIMessage):
                            continue
                        # Work on the message if it is a ToolMessage
                        # These may contain additional visuals that
                        # need to be displayed to the user.
                        # print("ToolMessage", msg)
                        # Skip the Tool message if it is an error message
                        if msg.status == "error":
                            continue

                        # Create a unique message id to identify the tool call
                        # msg.name is the name of the tool
                        # msg.tool_call_id is the unique id of the tool call
                        # st.session_state.run_id is the unique id of the run
                        uniq_msg_id = (
                            msg.name
                            + "_"
                            + msg.tool_call_id
                            + "_"
                            + str(st.session_state.run_id)
                        )
                        if msg.name in ["subgraph_extraction"]:
                            print(
                                "-",
                                len(current_state.values["dic_extracted_graph"]),
                                "subgraph_extraction",
                            )
                            # Add the graph into the visuals list
                            latest_graph = current_state.values["dic_extracted_graph"][
                                -1
                            ]
                            if current_state.values["dic_extracted_graph"]:
                                graphs_visuals.append(
                                    {
                                        "content": latest_graph["graph_dict"],
                                        "key": "subgraph_" + uniq_msg_id,
                                    }
                                )

            # Visualize the graph
            if len(graphs_visuals) > 0:
                for count, graph in enumerate(graphs_visuals):
                    streamlit_utils.render_graph(
                        graph_dict=graph["content"], key=graph["key"], save_graph=True
                    )

        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=streamlit_utils.submit_feedback,
                key=f"feedback_{st.session_state.run_id}",
            )
