#!/usr/bin/env python3

"""
Talk2Scholars: A Streamlit app for the Talk2Scholars graph.
"""

import os
import sys
import random
import streamlit as st
import hydra
import pandas as pd
from streamlit_feedback import streamlit_feedback
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.context import collect_runs
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client
from utils import streamlit_utils

sys.path.append("./")
from aiagents4pharma.talk2scholars.agents.main_agent import get_app

# Initialize configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
if "config" not in st.session_state:
    # Load Hydra configuration
    with hydra.initialize(
        version_base=None,
        config_path="../../aiagents4pharma/talk2scholars/configs",
    ):
        cfg = hydra.compose(config_name="config", overrides=["app/frontend=default"])
        cfg = cfg.app.frontend
        st.session_state.config = cfg
else:
    cfg = st.session_state.config

st.set_page_config(
    page_title=cfg.page.title, page_icon=cfg.page.icon, layout=cfg.page.layout
)
# Set the logo
st.logo(
    image="docs/assets/VPE.png",
    size="large",
    link="https://github.com/VirtualPatientEngine",
)


# Check if env variables OPENAI_API_KEY and/or NVIDIA_API_KEY exist
if cfg.api_keys.openai_key not in os.environ:
    st.error(
        "Please set the OPENAI_API_KEY "
        "environment variables in the terminal where you run "
        "the app. For more information, please refer to our "
        "[documentation](https://virtualpatientengine.github.io/AIAgents4Pharma/#option-2-git)."
    )
    st.stop()


# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Welcome to Talk2Scholars!"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = "Talk2Scholars-" + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    if "llm_model" not in st.session_state:
        st.session_state.app = get_app(
            st.session_state.unique_id,
            llm_model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        )
    else:
        print(st.session_state.llm_model)
        st.session_state.app = get_app(
            st.session_state.unique_id,
            llm_model=streamlit_utils.get_base_chat_model(st.session_state.llm_model),
        )
# Get the app
app = st.session_state.app


def _submit_feedback(user_response):
    """
    Function to submit feedback to the developers.
    """
    client = Client()
    client.create_feedback(
        st.session_state.run_id,
        key="feedback",
        score=1 if user_response["score"] == "👍" else 0,
        comment=user_response["text"],
    )
    st.info("Your feedback is on its way to the developers. Thank you!", icon="🚀")


# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write(
            """
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2Scholars
            </h3>
            """,
            unsafe_allow_html=True,
        )

        # LLM model panel
        st.selectbox(
            "Pick an LLM to power the agent",
            list(cfg.llms.available_models),
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model,
            help="Used for tool calling and generating responses.",
        )

        # Upload files (placeholder)
        # uploaded_file = st.file_uploader(
        #     "Upload sequencing data",
        #     accept_multiple_files=False,
        #     type=["h5ad"],
        #     help='''Upload a single h5ad file containing the sequencing data.
        #     The file should be in the AnnData format.'''
        #     )

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=575):
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
            elif message["type"] == "dataframe":
                if "tool_name" in message:
                    if message["tool_name"] in [
                        "display_results",
                    ]:
                        df_papers = message["content"]
                        st.dataframe(
                            df_papers,
                            use_container_width=True,
                            key=message["key"],
                            hide_index=True,
                            column_config={
                                "URL": st.column_config.LinkColumn(
                                    display_text="Open",
                                ),
                            },
                        )
                # else:
                #     streamlit_utils.render_table(message["content"],
                #                     key=message["key"],
                #                     # tool_name=message["tool_name"],
                #                     save_table=False)
                st.empty()

        # When the user asks a question
        if prompt:
            # Create a key 'uploaded_file' to read the uploaded file
            # if uploaded_file:
            #     st.session_state.article_pdf = uploaded_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append({"type": "message", "content": prompt_msg})
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

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
                        (
                            SystemMessage(content=m[1])
                            if m[0] == "system"
                            else (
                                HumanMessage(content=m[1])
                                if m[0] == "human"
                                else AIMessage(content=m[1])
                            )
                        )
                        for m in history
                    ]

                    # # Create config for the agent
                    config = {"configurable": {"thread_id": st.session_state.unique_id}}
                    # Update the LLM model
                    app.update_state(
                        config,
                        {
                            "llm_model": streamlit_utils.get_base_chat_model(
                                st.session_state.llm_model
                            )
                        },
                    )

                    streamlit_utils.get_response('T2S', None, app, st, prompt)

                    # # Create config for the agent
                    # config = {"configurable": {"thread_id": st.session_state.unique_id}}
                    # # Update the LLM model
                    # app.update_state(
                    #     config,
                    #     {
                    #         "llm_model": streamlit_utils.get_base_chat_model(
                    #             st.session_state.llm_model
                    #         )
                    #     },
                    # )
                    # # Update the agent state with the selected LLM model
                    # current_state = app.get_state(config)

                    # with collect_runs() as cb:
                    #     # Add Langsmith tracer
                    #     tracer = LangChainTracer(
                    #         project_name=st.session_state.project_name
                    #     )

                    #     # Get response from the agent with Langsmith tracing enabled
                    #     # response = app.invoke(
                    #     #     {"messages": [HumanMessage(content=prompt)]},
                    #     #     config=config | {"callbacks": [tracer]},
                    #     # )

                    #     response = app.stream(
                    #         {"messages": [HumanMessage(content=prompt)]},
                    #         config=config|{"callbacks": [tracer]},
                    #         stream_mode="messages"
                    #     )
                    #     st.write_stream(streamlit_utils.stream_response(response))

                    #     # Assign the traced run ID to session state
                    #     if cb.traced_runs:
                    #         st.session_state.run_id = cb.traced_runs[-1].id

                    # # # Get the latest agent state after the response
                    # # current_state = app.get_state(config)
                    # #
                    # # response = app.invoke(
                    # #     {"messages": [HumanMessage(content=prompt)]},
                    # #     config=config,
                    # # )

                    # current_state = app.get_state(config)

                    # # print (response["messages"])

                    # # Add assistant response to chat history
                    # assistant_msg = ChatMessage(
                    #     response["messages"][-1].content, role="assistant"
                    # )
                    # st.session_state.messages.append(
                    #     {"type": "message", "content": assistant_msg}
                    # )
                    # # Display the response in the chat
                    # st.markdown(response["messages"][-1].content)
                    # st.empty()
                    # reversed_messages = current_state.values["messages"][::-1]
                    # # Loop through the reversed messages until a
                    # # HumanMessage is found i.e. the last message
                    # # from the user. This is to display the results
                    # # of the tool calls made by the agent since the
                    # # last message from the user.
                    # for msg in reversed_messages:
                    #     # print (msg)
                    #     # Break the loop if the message is a HumanMessage
                    #     # i.e. the last message from the user
                    #     if isinstance(msg, HumanMessage):
                    #         break
                    #     # Skip the message if it is an AIMessage
                    #     # i.e. a message from the agent. An agent
                    #     # may make multiple tool calls before the
                    #     # final response to the user.
                    #     if isinstance(msg, AIMessage):
                    #         # print ('AIMessage', msg)
                    #         continue
                    #     # Work on the message if it is a ToolMessage
                    #     # These may contain additional visuals that
                    #     # need to be displayed to the user.
                    #     # print("ToolMessage", msg)
                    #     # Skip the Tool message if it is an error message
                    #     if msg.status == "error":
                    #         continue
                    #     # print("ToolMessage", msg)
                    #     uniq_msg_id = "_".join(
                    #         [msg.name, msg.tool_call_id, str(st.session_state.run_id)]
                    #     )
                        # if msg.name in ['search_tool',
                        #                 'get_single_paper_recommendations',
                        #                 'get_multi_paper_recommendations']:
                        # if msg.name in ["display_results"]:
                        #     # Display the results of the tool call
                        #     # for msg_artifact in msg.artifact:
                        #     # dic_papers = msg.artifact[msg_artifact]
                        #     dic_papers = msg.artifact
                        #     if not dic_papers:
                        #         continue
                        #     df_papers = pd.DataFrame.from_dict(
                        #         dic_papers, orient="index"
                        #     )
                        #     # Add index as a column "key"
                        #     df_papers["Key"] = df_papers.index
                        #     # Drop index
                        #     df_papers.reset_index(drop=True, inplace=True)
                        #     # Drop colum abstract
                        #     df_papers.drop(columns=["Abstract", "Key"], inplace=True)

                        #     if "Year" in df_papers.columns:
                        #         df_papers["Year"] = df_papers["Year"].apply(
                        #             lambda x: (
                        #                 str(int(x))
                        #                 if pd.notna(x) and str(x).isdigit()
                        #                 else None
                        #             )
                        #         )

                        #     if "Date" in df_papers.columns:
                        #         df_papers["Date"] = df_papers["Date"].apply(
                        #             lambda x: (
                        #                 pd.to_datetime(x, errors="coerce").strftime(
                        #                     "%Y-%m-%d"
                        #                 )
                        #                 if pd.notna(pd.to_datetime(x, errors="coerce"))
                        #                 else None
                        #             )
                        #         )

                        #     st.dataframe(
                        #         df_papers,
                        #         hide_index=True,
                        #         column_config={
                        #             "URL": st.column_config.LinkColumn(
                        #                 display_text="Open",
                        #             ),
                        #         },
                        #     )
                        #     # Add data to the chat history
                        #     st.session_state.messages.append(
                        #         {
                        #             "type": "dataframe",
                        #             "content": df_papers,
                        #             "key": "dataframe_" + uniq_msg_id,
                        #             "tool_name": msg.name,
                        #         }
                        #     )
                        #     st.empty()
        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=_submit_feedback,
                key=f"feedback_{st.session_state.run_id}",
            )
