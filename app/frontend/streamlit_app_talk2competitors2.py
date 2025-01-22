#!/usr/bin/env python3
"""
talk2comp: A Streamlit app for academic paper search and recommendations
"""

import os
import random
import sys

import streamlit as st
from langchain_core.messages import ChatMessage, HumanMessage

# from agents.main_agent import get_app
sys.path.append('./')
from aiagents4pharma.talk2competitors.agents.main_agent import get_app

# Page configuration
st.set_page_config(page_title="talk2comp", page_icon="📚", layout="wide")

# Styles for fixed bottom input
st.markdown(
    """
    <style>
        .stChatInput {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 1rem;
            background-color: #f2f3f5;
            z-index: 1000;
        }
        .main {
            margin-bottom: 100px;  # Space for fixed input
        }
        .stMarkdown {
            max-width: 100% !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "unique_id" not in st.session_state:
    st.session_state.unique_id = random.randint(1, 1000)
if "app" not in st.session_state:
    st.session_state.app = get_app(str(st.session_state.unique_id))

app = st.session_state.app

# Check OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Main layout
col1, col2 = st.columns([1, 4])

# Sidebar for settings
with col1:
    st.markdown("### 📚 talk2comp")
    llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    llm_option = st.selectbox(
        "Select LLM Model",
        llms,
        index=0,
    )

# Main chat area
with col2:
    # Container for chat history
    chat_container = st.container()
    with chat_container:
        st.markdown("### 💬 Chat History")

        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(
                message["content"].role,
                avatar="🤖" if message["content"].role != "user" else "👤",
            ):
                st.markdown(message["content"].content)

# Fixed bottom input
prompt = st.chat_input("Search for papers or ask questions...")

if prompt:
    # Display user message
    prompt_msg = ChatMessage(prompt, role="user")
    st.session_state.messages.append({"type": "message", "content": prompt_msg})

    # Get agent response
    with st.spinner("Processing your request..."):
        # Debug print before invoke
        print("Sending request to agent:", prompt)

        # Prepare initial state with all required fields
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "papers": [],
            "search_table": "",
            "next": None,
            "current_agent": None,
            "is_last_step": False,  # Ensure this is included
        }

        config = {"configurable": {"thread_id": str(st.session_state.unique_id)}}
        os.environ["AIAGENTS4PHARMA_LLM_MODEL"] = llm_option

        response = app.invoke(initial_state, config=config)

        # Debug print response
        print("Agent response:", response)

        # Add response to chat history and display
        if "messages" in response:
            # Get the last AI message
            ai_messages = [msg for msg in response["messages"] if msg.type == "ai"]
            if ai_messages:
                last_ai_message = ai_messages[-1]

                # Format paper results if present
                if "papers" in response and response["papers"]:
                    papers_content = response["papers"]
                    formatted_message = "Here are the papers I found:\n\n"

                    for idx, paper in enumerate(papers_content, start=1):
                        if isinstance(paper, str):
                            parts = paper.split("\n")
                            paper_id = parts[0].replace("Paper ID: ", "").strip()
                            title = parts[1].replace("Title: ", "").strip()
                            formatted_message += f"{idx}. **{title}**\n"
                            formatted_message += f"    - Paper ID: {paper_id}\n\n"
                else:
                    # Use the AI message content
                    formatted_message = last_ai_message.content

                assistant_msg = ChatMessage(formatted_message, role="assistant")
                st.session_state.messages.append(
                    {"type": "message", "content": assistant_msg}
                )

        # Rerun to update display
        st.rerun()
