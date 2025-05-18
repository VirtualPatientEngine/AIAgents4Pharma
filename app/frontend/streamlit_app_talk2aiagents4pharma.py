#!/usr/bin/env python3

'''
A Streamlit app for the Talk2AIAgents4Pharma graph.
'''

import os
import sys
import random
import hydra
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import streamlit_utils

st.set_page_config(page_title="Talk2AIAgents4Pharma",
                   page_icon="🤖",
                   layout="wide",
                   initial_sidebar_state="collapsed",)

# Set the logo
st.logo(
    image='docs/assets/VPE.png',
    size='large',
    link='https://github.com/VirtualPatientEngine'
)

# Check if env variables OPENAI_API_KEY and/or
# NVIDIA_API_KEY exist
if "OPENAI_API_KEY" not in os.environ or "NVIDIA_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY and NVIDIA_API_KEY "
             "environment variables in the terminal where you run "
             "the app. For more information, please refer to our "
             "[documentation](https://virtualpatientengine.github.io/AIAgents4Pharma/#option-2-git).")
    st.stop()

# Import the agent
sys.path.append('./')
from aiagents4pharma.talk2aiagents4pharma.agents.main_agent import get_app

# Initialize configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
if "config" not in st.session_state:
    # Load Hydra configuration
    with hydra.initialize(
        version_base=None,
        config_path="../../aiagents4pharma/talk2knowledgegraphs/configs",
    ):
        cfg_t2kg = hydra.compose(config_name="config", overrides=["app/frontend=default"])
        cfg_t2kg = cfg_t2kg.app.frontend
        st.session_state.config = cfg_t2kg
else:
    cfg_t2kg = st.session_state.config

########################################################################################
# Streamlit app
########################################################################################
# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", "Welcome to Talk2AIAgents4Pharma!"),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize current user
if "current_user" not in st.session_state:
    st.session_state.current_user = cfg_t2kg.default_user

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

## T2B

# Initialize sbml_file_path
if "sbml_file_path" not in st.session_state:
    st.session_state.sbml_file_path = None

## T2KG

# Initialize session state for pre-clinical data package uploader
if "data_package_key" not in st.session_state:
    st.session_state.data_package_key = 0

# Initialize session state for patient gene expression data uploader
if "endotype_key" not in st.session_state:
    st.session_state.endotype_key = 0

# Initialize session state for multimodal data package uploader
if "multimodal_key" not in st.session_state:
    st.session_state.multimodal_key = 0

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

    # Make directories if not exists
    os.makedirs(cfg_t2kg.upload_data_dir, exist_ok=True)

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = 'T2AA4P-' + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    if "llm_model" not in st.session_state:
        st.session_state.app = get_app(st.session_state.unique_id,
            llm_model=ChatOpenAI(model='gpt-4o-mini',
                       temperature=0))
    else:
        print (st.session_state.llm_model)
        st.session_state.app = get_app(st.session_state.unique_id,
                            llm_model=streamlit_utils.get_base_chat_model(
                                st.session_state.llm_model))

if "topk_nodes" not in st.session_state:
    # Subgraph extraction settings
    st.session_state.topk_nodes = cfg_t2kg.reasoning_subgraph_topk_nodes
    st.session_state.topk_edges = cfg_t2kg.reasoning_subgraph_topk_edges

# Get the app
app = st.session_state.app

# Apply custom CSS
streamlit_utils.apply_css()

# Sidebar
with st.sidebar:
    st.markdown("**⚙️ Subgraph Extraction Settings**")
    topk_nodes = st.slider(
        "Top-K (Nodes)",
        cfg_t2kg.reasoning_subgraph_topk_nodes_min,
        cfg_t2kg.reasoning_subgraph_topk_nodes_max,
        st.session_state.topk_nodes,
        key="st_slider_topk_nodes",
    )
    st.session_state.topk_nodes = topk_nodes
    topk_edges = st.slider(
        "Top-K (Edges)",
        cfg_t2kg.reasoning_subgraph_topk_nodes_min,
        cfg_t2kg.reasoning_subgraph_topk_nodes_max,
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
        st.write("""
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            Talk2AIAgents4Pharma
            </h3>
            """,
            unsafe_allow_html=True)

        # LLM model panel
        llms = ["OpenAI/gpt-4o-mini",
                "NVIDIA/llama-3.3-70b-instruct",
                "NVIDIA/llama-3.1-70b-instruct",
                "NVIDIA/llama-3.1-405b-instruct"]
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model,
            help="Used for tool calling and generating responses."
        )

        # Text embedding model panel
        text_models = ["NVIDIA/llama-3.2-nv-embedqa-1b-v2",
                       "OpenAI/text-embedding-ada-002"]
        st.selectbox(
            "Pick a text embedding model",
            text_models,
            index=0,
            key="text_embedding_model",
            on_change=streamlit_utils.update_text_embedding_model,
            kwargs={"app": app},
            help="Used for Retrival Augmented Generation (RAG) and other tasks."
        )

        # T2B Upload files
        uploaded_sbml_file = streamlit_utils.get_t2b_uploaded_files(app)

        # T2KG Upload files
        streamlit_utils.get_uploaded_files(cfg_t2kg)

        # Help text
        st.button("Know more ↗",
                #   icon="ℹ️",
                  on_click=streamlit_utils.help_button,
                  use_container_width=False)

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=1200):
        st.write("#### 💬 Chat History")

        # Display history of messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(message["content"].role,
                                     avatar="🤖"
                                     if message["content"].role != 'user'
                                     else "👩🏻‍💻"):
                    st.markdown(message["content"].content)
                    st.empty()
            elif message["type"] == "button":
                if st.button(message["content"],
                             key=message["key"]):
                    # Trigger the question
                    prompt = message["question"]
                    st.empty()
            elif message["type"] == "plotly":
                streamlit_utils.render_plotly(message["content"],
                              key=message["key"],
                              title=message["title"],
                              y_axis_label=message["y_axis_label"],
                              x_axis_label=message["x_axis_label"],
                            #   tool_name=message["tool_name"],
                              save_chart=False)
                st.empty()
            elif message["type"] == "toggle":
                streamlit_utils.render_toggle(key=message["key"],
                                    toggle_text=message["content"],
                                    toggle_state=message["toggle_state"],
                                    save_toggle=False)
                st.empty()
            elif message["type"] == "graph":
                streamlit_utils.render_graph(
                    message["content"], key=message["key"], save_graph=False
                )
                st.empty()
            elif message["type"] == "dataframe":
                if 'tool_name' in message:
                    if message['tool_name'] == 'get_annotation':
                        df_selected = message["content"]
                        st.dataframe(df_selected,
                                    use_container_width=True,
                                    key=message["key"],
                                    hide_index=True,
                                    column_config={
                                        "Id": st.column_config.LinkColumn(
                                            label="Id",
                                            help="Click to open the link associated with the Id",
                                            validate=r"^http://.*$",  # Ensure the link is valid
                                            display_text=r"^http://identifiers\.org/(.*?)$"
                                        ),
                                        "Species Name": st.column_config.TextColumn("Species Name"),
                                        "Description": st.column_config.TextColumn("Description"),
                                        "Database": st.column_config.TextColumn("Database"),
                                    }
                                )
                    elif message['tool_name'] == 'search_models':
                        df_selected = message["content"]
                        st.dataframe(df_selected,
                            use_container_width=True,
                            key=message["key"],
                            hide_index=True,
                            column_config={
                                "url": st.column_config.LinkColumn(
                                    label="ID",
                                    help="Click to open the link associated with the Id",
                                    validate=r"^http://.*$",  # Ensure the link is valid
                                    display_text=r"^https://www.ebi.ac.uk/biomodels/(.*?)$"
                                ),
                                "name": st.column_config.TextColumn("Name"),
                                "format": st.column_config.TextColumn("Format"),
                                "submissionDate": st.column_config.TextColumn("Submission Date"),
                                }
                            )
                else:
                    streamlit_utils.render_table(message["content"],
                                    key=message["key"],
                                    # tool_name=message["tool_name"],
                                    save_table=False)
                st.empty()
        # Display intro message only the first time
        # i.e. when there are no messages in the chat
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Initializing the agent ..."):
                    config = {"configurable":
                                {"thread_id": st.session_state.unique_id}
                                }
                    # Update the agent state with the selected LLM model
                    current_state = app.get_state(config)
                    app.update_state(
                        config,
                        {"llm_model": streamlit_utils.get_base_chat_model(
                            st.session_state.llm_model),
                        "text_embedding_model": streamlit_utils.get_text_embedding_model(
                            st.session_state.text_embedding_model),
                        "embedding_model": streamlit_utils.get_text_embedding_model(
                            st.session_state.text_embedding_model),
                        "uploaded_files": st.session_state.uploaded_files,
                        "topk_nodes": st.session_state.topk_nodes,
                        "topk_edges": st.session_state.topk_edges,
                        "dic_source_graph": [
                            {
                                "name": st.session_state.config["kg_name"],
                                "kg_pyg_path": st.session_state.config["kg_pyg_path"],
                                "kg_text_path": st.session_state.config["kg_text_path"],
                            }
                        ]}
                    )
                    intro_prompt = "Tell your name and about yourself. Always start with a greeting."
                    intro_prompt += " and tell about the tools you can run to perform analysis with short description."
                    intro_prompt += " We have provided starter questions (separately) outisde your response."
                    intro_prompt += " Do not provide any questions by yourself. Let the users know that they can"
                    intro_prompt += " simply click on the questions to execute them."
                    # intro_prompt += " Let them know that they can check out the use cases"
                    # intro_prompt += " and FAQs described in the link below. Be friendly and helpful."
                    # intro_prompt += "\n"
                    # intro_prompt += "Here is the link to the use cases: [Use Cases](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2biomodels/cases/Case_1/)"
                    # intro_prompt += "\n"
                    # intro_prompt += "Here is the link to the FAQs: [FAQs](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2biomodels/faq/)"
                    response = app.stream(
                                    {"messages": [HumanMessage(content=intro_prompt)]},
                                    config=config,
                                    stream_mode="messages"
                                )
                    st.write_stream(streamlit_utils.stream_response(response))
                    current_state = app.get_state(config)
                    # Add response to chat history
                    assistant_msg = ChatMessage(
                                        current_state.values["messages"][-1].content,
                                        role="assistant")
                    st.session_state.messages.append({
                                    "type": "message",
                                    "content": assistant_msg
                                })
                    st.empty()
        if len(st.session_state.messages) <= 1:
            for count, question in enumerate(streamlit_utils.sample_questions_t2aa4p()):
                if st.button(f'Q{count+1}. {question}',
                             key=f'sample_question_{count+1}'):
                    # Trigger the question
                    prompt = question
                # Add button click to chat history
                st.session_state.messages.append({
                                "type": "button",
                                "question": question,
                                "content": f'Q{count+1}. {question}',
                                "key": f'sample_question_{count+1}'
                            })

        # When the user asks a question
        if prompt:
            # Create a key 'uploaded_file' to read the uploaded file
            if uploaded_sbml_file:
                st.session_state.sbml_file_path = uploaded_sbml_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append(
                {
                    "type": "message",
                    "content": prompt_msg
                }
            )
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

            # Auxiliary visualization-related variables
            graphs_visuals = []
            with st.chat_message("assistant", avatar="🤖"):
                # with st.spinner("Fetching response ..."):
                with st.spinner():
                    # Get chat history
                    history = [(m["content"].role, m["content"].content)
                                            for m in st.session_state.messages
                                            if m["type"] == "message"]
                    # Convert chat history to ChatMessage objects
                    chat_history = [
                        SystemMessage(content=m[1]) if m[0] == "system" else
                        HumanMessage(content=m[1]) if m[0] == "human" else
                        AIMessage(content=m[1])
                        for m in history
                    ]

                    streamlit_utils.get_response('T2AA4P',
                                                 graphs_visuals,
                                                 app,
                                                 st,
                                                 prompt)
            # Visualize the graph
            if len(graphs_visuals) > 0:
                for count, graph in enumerate(graphs_visuals):
                    streamlit_utils.render_graph(
                        graph_dict=graph["content"], key=graph["key"], save_graph=True
                    )

        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=streamlit_utils.submit_feedback,
                key=f"feedback_{st.session_state.run_id}"
            )
