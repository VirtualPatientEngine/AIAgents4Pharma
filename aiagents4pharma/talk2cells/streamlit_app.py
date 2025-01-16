#!/usr/bin/env python3

'''
Talk2Cells: A Streamlit app for the Talk2Cells graph.
'''

from langchain_core.messages import HumanMessage
from agents.agent_scp import get_app
import os
import streamlit as st
import random
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Talk2Cells", page_icon="🤖", layout="wide")

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages([
        ("system", "Welcome to Talk2Cells!"),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize graph
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    st.session_state.app = get_app(st.session_state.unique_id)

# Get the app
app = st.session_state.app

# Check if env variable OPENAI_API_KEY exists
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment \
        variable in the terminal where you run the app.")
    st.stop()

# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write("""
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2Cells
            </h3>
            """,
            unsafe_allow_html=True)

        # LLM panel (Only at the front-end for now)
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        llm_option = st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="st_selectbox_llm"
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
            with st.chat_message(message["content"].role,
                                    avatar="🤖" 
                                    if message["content"].role != 'user'
                                    else "👩🏻‍💻"):
                st.markdown(message["content"].content)
                st.empty()

        # When the user asks a question
        if prompt:
            # Create a key 'uploaded_file' to read the uploaded file
            # if uploaded_file:
            #     st.session_state.article_pdf = uploaded_file.read().decode("utf-8")

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
                    # app.update_state(config, {"llm_model": llm_option})
                    current_state = app.get_state(config)
                    # st.markdown(current_state.values["llm_model"])

                    # Set the environment variable AIAGENTS4PHARMA_LLM_MODEL
                    os.environ["AIAGENTS4PHARMA_LLM_MODEL"] = llm_option

                    # Get response from the agent
                    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
                    # Print the response
                    # print (response)

                    # Add assistant response to chat history
                    assistant_msg = ChatMessage(response["messages"][-1].content,
                                                role="assistant")
                    st.session_state.messages.append({
                                    "type": "message",
                                    "content": assistant_msg
                                })
                    # Display the response in the chat
                    st.markdown(response["messages"][-1].content)
                    st.empty()
