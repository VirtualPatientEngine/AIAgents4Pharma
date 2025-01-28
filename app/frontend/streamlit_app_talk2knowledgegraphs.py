#!/usr/bin/env python3

'''
Talk2KnowledgeGraphs: A Streamlit app for the Talk2KnowledgeGraphs graph.
'''

import os
import sys
import random
import streamlit as st
import pandas as pd
from streamlit_feedback import streamlit_feedback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tracers.context import collect_runs
from langchain.callbacks.tracers import LangChainTracer
from utils import streamlit_utils
sys.path.append('./')
from aiagents4pharma.talk2knowledgegraphs.agents.t2kg_agent import get_app
# from talk2knowledgegraphs.agents.t2kg_agent import get_app

st.set_page_config(page_title="Talk2KnowledgeGraphs", page_icon="🤖", layout="wide")


# st.logo(
#     image='docs/VPE.png',
#     size='large',
#     link='https://github.com/VirtualPatientEngine'
# )

# Check if env variable OPENAI_API_KEY exists
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment \
        variable in the terminal where you run the app.")
    st.stop()

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", "Welcome to Talk2KnowledgeGraphs!"),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize documents_path
if "documents_path" not in st.session_state:
    st.session_state.documents_path = None

# Initialize project_name for Langsmith
if "project_name" not in st.session_state:
    # st.session_state.project_name = str(st.session_state.user_name) + '@' + str(uuid.uuid4())
    st.session_state.project_name = 'T2KG-' + str(random.randint(1000, 9999))

# Initialize run_id for Langsmith
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    if "llm_model" not in st.session_state:
        st.session_state.app = get_app(st.session_state.unique_id)
    else:
        st.session_state.app = get_app(st.session_state.unique_id,
                                       llm_model=st.session_state.llm_model)

# Get the app
app = st.session_state.app

# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write("""
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2KnowledgeGraphs
            </h3>
            """,
            unsafe_allow_html=True)

        # LLM panel (Only at the front-end for now)
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model
        )

        # Upload files (placeholder)
        doc_uploaded_file = st.file_uploader(
            "Upload additional document(s)",
            accept_multiple_files=False,
            type=["pdf"],
            help='''Upload document(s) as additional information for the agent to utilize.'''
            )

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
    with st.container(border=True, height=575):
        st.write("#### 💬 Chat History")

        # Display chat messages
        for count, message in enumerate(st.session_state.messages):
            if message["type"] == "message":
                with st.chat_message(message["content"].role,
                                     avatar="🤖"
                                     if message["content"].role != 'user'
                                     else "👩🏻‍💻"):
                    st.markdown(message["content"].content)
                    st.empty()
            elif message["type"] == "plotly":
                streamlit_utils.render_plotly(message["content"],
                              key=message["key"],
                              title=message["title"],
                            #   tool_name=message["tool_name"],
                              save_chart=False)
                st.empty()
            elif message["type"] == "toggle":
                streamlit_utils.render_toggle(key=message["key"],
                                    toggle_text=message["content"],
                                    toggle_state=message["toggle_state"],
                                    save_toggle=False)
                st.empty()
            elif message["type"] == "dataframe":
                streamlit_utils.render_table(message["content"],
                                key=message["key"],
                                # tool_name=message["tool_name"],
                                save_table=False)
                st.empty()
            elif message["type"] == "graph":
                streamlit_utils.render_graph(message["content"],
                                key=message["key"],
                                save_graph=False)
                st.empty()

        # When the user asks a question
        if prompt:
            # Create a key 'doc_uploaded_file' to read the uploaded file
            if doc_uploaded_file:
                st.session_state.documents_path = doc_uploaded_file.read().decode("utf-8")

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

                    # Create config for the agent
                    config = {"configurable": {"thread_id": st.session_state.unique_id}}
                    # Update the agent state with the selected LLM model
                    current_state = app.get_state(config)
                    app.update_state(
                        config,
                        {"documents_path": [st.session_state.documents_path]}
                    )
                    app.update_state(
                        config,
                        {"llm_model": st.session_state.llm_model}
                    )
                    # print (current_state.values)
                    # current_state = app.get_state(config)
                    # print ('updated state', current_state.values["documents_path"])

                    ERROR_FLAG = False
                    with collect_runs() as cb:
                        # Add Langsmith tracer
                        tracer = LangChainTracer(project_name=st.session_state.project_name)
                        # Get response from the agent
                        response = app.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config|{"callbacks": [tracer]}
                        )
                        st.session_state.run_id = cb.traced_runs[-1].id
                    # print(response["messages"])
                    current_state = app.get_state(config)
                    # print (current_state.values["model_id"])

                    # Add response to chat history
                    assistant_msg = ChatMessage(
                                        response["messages"][-1].content,
                                        role="assistant")
                    st.session_state.messages.append({
                                    "type": "message",
                                    "content": assistant_msg
                                })
                    # Display the response in the chat
                    st.markdown(response["messages"][-1].content)
                    st.empty()

                    # Get the current state of the graph
                    current_state = app.get_state(config)

                    # Add the graph into the visuals list
                    if current_state.values["graph_dict"]:
                        graphs_visuals.append(current_state.values["graph_dict"])

            # Visualize the graph
            if len(graphs_visuals) > 0:
                for count, graph in enumerate(graphs_visuals):
                    streamlit_utils.render_graph(
                        graph_dict=current_state.values["graph_dict"],
                        key="subgraph",
                        save_graph=True
                    )

                    # print('current_state', current_state.values)
                    # print('current_state', current_state.values["graph_text"])
                    # print('current_state', current_state.values["graph_text"])

                    # # Get the messages from the current state
                    # # and reverse the order
                    # reversed_messages = current_state.values["messages"][::-1]
                    # print(reversed_messages)

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
                    #         continue
                    #     # Work on the message if it is a ToolMessage
                    #     # These may contain additional visuals that
                    #     # need to be displayed to the user.
                    #     print("ToolMessage", msg)
                    #     # Skip the Tool message if it is an error message
                    #     if msg.status == "error":
                    #         continue

                    #     # Create a unique message id to identify the tool call
                    #     # msg.name is the name of the tool
                    #     # msg.tool_call_id is the unique id of the tool call
                    #     # st.session_state.run_id is the unique id of the run
                    #     uniq_msg_id = msg.name+'_'+msg.tool_call_id+'_'+str(st.session_state.run_id)
                    #     if msg.name in ["simulate_model", "custom_plotter"]:
                    #         if msg.name == "simulate_model":
                    #             # Convert the simulated data to a single dictionary
                    #             dic_simulated_data = {}
                    #             for data in current_state.values["dic_simulated_data"]:
                    #                 for key in data:
                    #                     if key not in dic_simulated_data:
                    #                         dic_simulated_data[key] = []
                    #                     dic_simulated_data[key] += [data[key]]
                    #             # Create a pandas dataframe from the dictionary
                    #             df_simulated_data = pd.DataFrame.from_dict(dic_simulated_data)
                    #             # Get the simulated data for the current tool call
                    #             df_simulated = pd.DataFrame(
                    #                 df_simulated_data[df_simulated_data['tool_call_id'] == msg.tool_call_id]['data'].iloc[0])
                    #             df_selected = df_simulated
                    #         else:
                    #             if msg.artifact:
                    #                 df_selected = pd.DataFrame.from_dict(msg.artifact)
                    #                 # print (df_selected)
                    #             else:
                    #                 continue
                    #         # Display the toggle button to suppress the table
                    #         streamlit_utils.render_table_plotly(
                    #             uniq_msg_id, msg.content, df_selected)
                    #     elif msg.name == "parameter_scan":
                    #         # Convert the scanned data to a single dictionary
                    #         print ('-', len(current_state.values["dic_scanned_data"]))
                    #         dic_scanned_data = {}
                    #         for data in current_state.values["dic_scanned_data"]:
                    #             print ('-', data['name'])
                    #             for key in data:
                    #                 if key not in dic_scanned_data:
                    #                     dic_scanned_data[key] = []
                    #                 dic_scanned_data[key] += [data[key]]
                    #         # Create a pandas dataframe from the dictionary
                    #         df_scanned_data = pd.DataFrame.from_dict(dic_scanned_data)
                    #         # Get the scanned data for the current tool call
                    #         df_scanned_current_tool_call = pd.DataFrame(
                    #             df_scanned_data[df_scanned_data['tool_call_id'] == msg.tool_call_id])
                    #         # df_scanned_current_tool_call.drop_duplicates()
                    #         # print (df_scanned_current_tool_call)
                    #         for count in range(0, len(df_scanned_current_tool_call.index)):
                    #             # Get the scanned data for the current tool call
                    #             df_selected = pd.DataFrame(
                    #                 df_scanned_data[df_scanned_data['tool_call_id'] == msg.tool_call_id]['data'].iloc[count])
                    #             # Display the toggle button to suppress the table
                    #             streamlit_utils.render_table_plotly(
                    #             uniq_msg_id+'_'+str(count),
                    #             df_scanned_current_tool_call['name'].iloc[count],
                    #             df_selected)

        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=streamlit_utils.submit_feedback,
                key=f"feedback_{st.session_state.run_id}"
            )
