#!/usr/bin/env python3

"""
Talk2BioModels: Interactive BioModel Simulation Tool
"""

import os
import random
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append("./")
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from aiagents4pharma.talk2biomodels.tools.ask_question import AskQuestionTool
from aiagents4pharma.talk2biomodels.tools.custom_plotter import CustomPlotterTool
from aiagents4pharma.talk2biomodels.tools.fetch_parameters import FetchParametersTool
from aiagents4pharma.talk2biomodels.tools.get_annotation import GetAnnotationTool
from aiagents4pharma.talk2biomodels.tools.model_description import ModelDescriptionTool
from aiagents4pharma.talk2biomodels.tools.search_models import SearchModelsTool
from aiagents4pharma.talk2biomodels.tools.simulate_model import SimulateModelTool

# Set the streamlit session key for the sys bio model
ST_SYS_BIOMODEL_KEY = "last_model_object"
ST_SESSION_DF = "last_annotations_df"

st.set_page_config(page_title="Talk2BioModels", page_icon="🤖", layout="wide")
st.logo(
    image="./app/frontend/VPE.png", link="https://www.github.com/virtualpatientengine"
)

# Define tools and their metadata
simulate_model = SimulateModelTool(st_session_key=ST_SYS_BIOMODEL_KEY)
ask_question = AskQuestionTool(st_session_key=ST_SYS_BIOMODEL_KEY)
with open("./app/frontend/prompts/prompt_ask_question.txt", encoding="utf-8") as file:
    prompt_content = file.read()
ask_question.metadata = {"prompt": prompt_content}
# plot_figure = PlotImageTool(st_session_key=ST_SYS_BIOMODEL_KEY)
model_description = ModelDescriptionTool(st_session_key=ST_SYS_BIOMODEL_KEY)
with open(
    "./app/frontend/prompts/prompt_model_description.txt", encoding="utf-8"
) as file:
    prompt_content = file.read()
model_description.metadata = {"prompt": prompt_content}
search_models = SearchModelsTool()
custom_plotter = CustomPlotterTool(st_session_key=ST_SYS_BIOMODEL_KEY)
fetch_parameters = FetchParametersTool(st_session_key=ST_SYS_BIOMODEL_KEY)
get_annotation = GetAnnotationTool(
    st_session_key=ST_SYS_BIOMODEL_KEY, st_session_df=ST_SESSION_DF
)

tools = [
    simulate_model,
    ask_question,
    #  plot_figure,
    custom_plotter,
    fetch_parameters,
    model_description,
    search_models,
    get_annotation,
]

# Load the prompt for the main agent
with open("./app/frontend/prompts/prompt_general.txt", encoding="utf-8") as file:
    prompt_content = file.read()

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_content),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the OpenAI model
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o-mini",
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create an agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)


def render_plotly(df_simulation_results: pd.DataFrame) -> px.line:
    """
    Function to visualize the dataframe using Plotly.

    Args:
        df: pd.DataFrame: The input dataframe
    """
    df_simulation_results = df_simulation_results.melt(
        id_vars="Time", var_name="Parameters", value_name="Concentration"
    )
    fig = px.line(
        df_simulation_results,
        x="Time",
        y="Concentration",
        color="Parameters",
        title="Concentration of parameters over time",
        height=500,
        width=600,
    )
    return fig


def get_random_spinner_text():
    """
    Function to get a random spinner text.
    """
    spinner_texts = [
        "Your request is being carefully prepared. one moment, please.",
        "Working on that for you now—thanks for your patience.",
        "Hold tight! I’m getting that ready for you.",
        "I’m on it! Just a moment, please.",
        "Running algorithms... your answer is on its way.",
        "Processing your request. Please hold on...",
        "One moment while I work on that for you...",
        "Fetching the details for you. This won’t take long.",
        "Sit back while I take care of this for you.",
    ]
    return random.choice(spinner_texts)


# Main layout of the app split into two columns
main_col1, main_col2 = st.columns([3, 7])
# First column
with main_col1:
    with st.container(border=True):
        # Title
        st.write(
            """
            <h3 style='margin: 0px; padding-bottom: 10px; font-weight: bold;'>
            🤖 Talk2BioModels
            </h3>
            """,
            unsafe_allow_html=True,
        )

        # LLM panel
        llms = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        llm_option = st.selectbox(
            "Pick an LLM to power the agent", llms, index=0, key="st_selectbox_llm"
        )

        # Upload files
        uploaded_file = st.file_uploader(
            "Upload an XML/SBML file",
            accept_multiple_files=False,
            type=["xml", "sbml"],
            help="""Upload an XML/SBML file to simulate a biological model, \
                and ask questions about the simulation results.""",
        )

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
            elif message["type"] == "plotly":
                st.plotly_chart(
                    render_plotly(message["content"]),
                    use_container_width=True,
                    key=f"plotly_{count}",
                )
            elif message["type"] == "dataframe":
                st.dataframe(
                    message["content"],
                    use_container_width=True,
                    key=f"dataframe_{count}",
                )
        if prompt:
            if ST_SYS_BIOMODEL_KEY not in st.session_state:
                st.session_state[ST_SYS_BIOMODEL_KEY] = None

            if ST_SESSION_DF not in st.session_state:
                st.session_state[ST_SESSION_DF] = None

            # Create a key 'uploaded_file' to read the uploaded file
            if uploaded_file:
                st.session_state.sbml_file_path = uploaded_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(prompt, role="user")
            st.session_state.messages.append({"type": "message", "content": prompt_msg})
            with st.chat_message("user", avatar="👩🏻‍💻"):
                st.markdown(prompt)
                st.empty()

            with st.chat_message("assistant", avatar="🤖"):
                # with st.spinner("Fetching response ..."):
                with st.spinner(get_random_spinner_text()):
                    history = [
                        (m["content"].role, m["content"].content)
                        for m in st.session_state.messages
                        if m["type"] == "message"
                    ]
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
                    # Call the agent
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": chat_history}
                    )

                    # Ensure response["output"] is a valid string
                    output_content = response.get("output", "")

                    # If output is a dictionary (like an error message), handle it properly
                    if isinstance(output_content, dict):
                        # Extract error message or default message
                        output_content = str(
                            output_content.get("error", "Unknown error occurred")
                        )

                    # Add assistant response to chat history
                    assistant_msg = ChatMessage(
                        content=output_content, role="assistant"
                    )
                    st.session_state.messages.append(
                        {"type": "message", "content": assistant_msg}
                    )

                    # Display the response
                    st.markdown(output_content)
                    st.empty()
                    print(response)
                    if (
                        "intermediate_steps" in response
                        and len(response["intermediate_steps"]) > 0
                    ):
                        for r in response["intermediate_steps"]:
                            # Inside the agent_executor chain:
                            if r[0].tool == "get_annotation":
                                annotations_df = st.session_state[ST_SESSION_DF]
                                # Display the DataFrame in Streamlit frontend
                                st.dataframe(annotations_df, use_container_width=True)
                                # Append the DataFrame to chat history (if necessary)
                                st.session_state.messages.append(
                                    {"type": "dataframe", "content": annotations_df}
                                )

                            elif r[0].tool == "simulate_model":
                                model_obj = st.session_state[ST_SYS_BIOMODEL_KEY]
                                df_sim_results = model_obj.simulation_results
                                # Add data to the chat history
                                st.session_state.messages.append(
                                    {"type": "dataframe", "content": df_sim_results}
                                )
                                st.dataframe(df_sim_results, use_container_width=True)
                                # Add the plotly chart to the chat history
                                st.session_state.messages.append(
                                    {"type": "plotly", "content": df_sim_results}
                                )
                                # Display the plotly chart
                                st.plotly_chart(
                                    render_plotly(df_sim_results),
                                    use_container_width=True,
                                )

                            elif r[0].tool == "custom_plotter":
                                model_obj = st.session_state[ST_SYS_BIOMODEL_KEY]
                                # Prepare df_subset for custom_simulation_results
                                df_subset = pd.DataFrame()
                                if len(st.session_state.custom_simulation_results) > 0:
                                    custom_headers = (
                                        st.session_state.custom_simulation_results
                                    )
                                    custom_headers = list(custom_headers)
                                    # Add Time column to the custom headers
                                    if "Time" not in custom_headers:
                                        custom_headers = ["Time"] + custom_headers

                                    # Make df_subset with only the custom headers
                                    df_subset = model_obj.simulation_results[
                                        custom_headers
                                    ]
                                    # Add data to the chat history
                                    st.session_state.messages.append(
                                        {"type": "dataframe", "content": df_subset}
                                    )
                                    st.dataframe(df_subset, use_container_width=True)
                                    # Add the plotly chart to the chat history
                                    st.session_state.messages.append(
                                        {"type": "plotly", "content": df_subset}
                                    )
                                    # Display the plotly chart
                                    st.plotly_chart(
                                        render_plotly(df_subset),
                                        use_container_width=True,
                                    )
                    else:
                        # If intermediate_steps is empty, show a message
                        st.warning("No intermediate steps were found in the response.")
