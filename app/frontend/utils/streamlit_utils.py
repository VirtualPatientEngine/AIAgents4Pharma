#!/usr/bin/env python3

"""
Utils for Streamlit.
"""

import datetime
import os
import pickle
import sys
import tempfile

import gravis
import hydra
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ChatMessage, HumanMessage
from langchain_core.tracers.context import collect_runs
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith import Client

sys.path.append("./")
import aiagents4pharma.talk2scholars.tools.pdf.question_and_answer as qa_module
from aiagents4pharma.talk2scholars.tools.pdf.utils.generate_answer import (
    load_hydra_config,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store import Vectorstore
from aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper import (
    ZoteroSearchData,
)


def submit_feedback(user_response):
    """
    Function to submit feedback to the developers.

    Args:
        user_response: dict: The user response
    """
    client = Client()
    client.create_feedback(
        st.session_state.run_id,
        key="feedback",
        score=1 if user_response["score"] == "👍" else 0,
        comment=user_response["text"],
    )
    st.info("Your feedback is on its way to the developers. Thank you!", icon="🚀")


def render_table_plotly(
    uniq_msg_id, content, df_selected, x_axis_label="Time", y_axis_label="Concentration"
):
    """
    Function to render the table and plotly chart in the chat.

    Args:
        uniq_msg_id: str: The unique message id
        msg: dict: The message object
        df_selected: pd.DataFrame: The selected dataframe
    """
    # Display the toggle button to suppress the table
    render_toggle(
        key="toggle_plotly_" + uniq_msg_id,
        toggle_text="Show Plot",
        toggle_state=True,
        save_toggle=True,
    )
    # Display the plotly chart
    render_plotly(
        df_selected,
        key="plotly_" + uniq_msg_id,
        title=content,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
        save_chart=True,
    )
    # Display the toggle button to suppress the table
    render_toggle(
        key="toggle_table_" + uniq_msg_id,
        toggle_text="Show Table",
        toggle_state=False,
        save_toggle=True,
    )
    # Display the table
    render_table(df_selected, key="dataframe_" + uniq_msg_id, save_table=True)
    st.empty()


def render_toggle(
    key: str, toggle_text: str, toggle_state: bool, save_toggle: bool = False
):
    """
    Function to render the toggle button to show/hide the table.

    Args:
        key: str: The key for the toggle button
        toggle_text: str: The text for the toggle button
        toggle_state: bool: The state of the toggle button
        save_toggle: bool: Flag to save the toggle button to the chat history
    """
    st.toggle(toggle_text, toggle_state, help="""Toggle to show/hide data""", key=key)
    # print (key)
    if save_toggle:
        # Add data to the chat history
        st.session_state.messages.append(
            {
                "type": "toggle",
                "content": toggle_text,
                "toggle_state": toggle_state,
                "key": key,
            }
        )


def render_plotly(
    df: pd.DataFrame,
    key: str,
    title: str,
    y_axis_label: str,
    x_axis_label: str,
    save_chart: bool = False,
):
    """
    Function to visualize the dataframe using Plotly.

    Args:
        df: pd.DataFrame: The input dataframe
        key: str: The key for the plotly chart
        title: str: The title of the plotly chart
        save_chart: bool: Flag to save the chart to the chat history
    """
    # toggle_state = st.session_state[f'toggle_plotly_{tool_name}_{key.split("_")[-1]}']\
    toggle_state = st.session_state[f'toggle_plotly_{key.split("plotly_")[1]}']
    if toggle_state:
        df_simulation_results = df.melt(
            id_vars="Time", var_name="Species", value_name="Concentration"
        )
        fig = px.line(
            df_simulation_results,
            x="Time",
            y="Concentration",
            color="Species",
            title=title,
            height=500,
            width=600,
        )
        # Set y axis label
        fig.update_yaxes(title_text=f"Quantity ({y_axis_label})")
        # Set x axis label
        fig.update_xaxes(title_text=f"Time ({x_axis_label})")
        # Display the plotly chart
        st.plotly_chart(fig, use_container_width=True, key=key)
    if save_chart:
        # Add data to the chat history
        st.session_state.messages.append(
            {
                "type": "plotly",
                "content": df,
                "key": key,
                "title": title,
                "y_axis_label": y_axis_label,
                "x_axis_label": x_axis_label,
                # "tool_name": tool_name
            }
        )


def render_table(df: pd.DataFrame, key: str, save_table: bool = False):
    """
    Function to render the table in the chat.

    Args:
        df: pd.DataFrame: The input dataframe
        key: str: The key for the table
        save_table: bool: Flag to save the table to the chat history
    """
    # print (st.session_state['toggle_simulate_model_'+key.split("_")[-1]])
    # toggle_state = st.session_state[f'toggle_table_{tool_name}_{key.split("_")[-1]}']
    toggle_state = st.session_state[f'toggle_table_{key.split("dataframe_")[1]}']
    if toggle_state:
        st.dataframe(df, use_container_width=True, key=key)
    if save_table:
        # Add data to the chat history
        st.session_state.messages.append(
            {
                "type": "dataframe",
                "content": df,
                "key": key,
                # "tool_name": tool_name
            }
        )


def sample_questions():
    """
    Function to get the sample questions.
    """
    questions = [
        'Search for all biomodels on "Crohns Disease"',
        "Briefly describe biomodel 971 and simulate it for 50 days with an interval of 50.",
        "Bring biomodel 27 to a steady state, and then "
        "determine the Mpp concentration at the steady state.",
        "How will the concentration of Mpp change in model 27, "
        "if the initial value of MAPKK were to be changed between 1 and 100 in steps of 10?",
        "Show annotations of all interleukins in model 537",
    ]
    return questions


def sample_questions_t2s():
    """
    Function to get the sample questions for Talk2Scholars.
    """
    questions = [
        "Find articles on 'Bridging Biomedical Foundation Models via Knowledge Graphs'.",
        "Tell me more about the first article in the last search results",
        "Save these articles in my Zotero library under the collection 'Curiosity'",
        "Download the last displayed articles and summarize the pre-trained foundation models used in the articles.",
        "Show all the papers in my Zotero library.",
        "Describe the PDB IDs of the GPCR 3D structures used in all the PDFs, and explain how the embeddings of the GPCR sequences were generated.",
    ]
    return questions


def initialize_zotero_and_build_store():
    """
    Download all PDFs from the user's Zotero library and build a RAG vector store.
    """
    # Retrieve the agent app from session state
    app = st.session_state.app
    # Fetch Zotero items and download PDFs
    search_data = ZoteroSearchData(
        query="",
        only_articles=True,
        limit=1,
        tool_call_id="startup",
        download_pdfs=True,
    )
    search_data.process_search()
    results = search_data.get_search_results()
    # Save article metadata and PDF paths
    st.session_state.article_data = results.get("article_data", {})
    # Update agent state with article data
    config = {"configurable": {"thread_id": st.session_state.unique_id}}
    app.update_state(config, {"article_data": st.session_state.article_data})
    # Build RAG vector store
    pdf_config = load_hydra_config()
    embedding_model = get_text_embedding_model(st.session_state.text_embedding_model)
    vector_store = Vectorstore(embedding_model=embedding_model, config=pdf_config)
    for paper_id, meta in st.session_state.article_data.items():
        pdf_url = meta.get("pdf_url")
        if pdf_url:
            vector_store.add_paper(paper_id, pdf_url, meta)
    vector_store.build_vector_store()
    # Expose the vector store for use by the Q&A tool helper
    # (helper.prebuilt_vector_store caches the shared store)
    qa_module.helper.prebuilt_vector_store = vector_store
    # Mark as initialized to prevent rerunning
    st.session_state.zotero_initialized = True


def sample_questions_t2aa4p():
    """
    Function to get the sample questions for Talk2AIAgents4Pharma.
    """
    questions = [
        'Search for all the biomodels on "Crohns Disease"',
        "Briefly describe biomodel 537 and simulate it for 2016 hours with an interval of 100.",
        "List the drugs that target Interleukin-6",
        "What genes are associated with Crohn's disease?",
    ]
    return questions


def stream_response(response):
    """
    Function to stream the response from the agent.

    Args:
        response: dict: The response from the agent
    """
    agent_responding = False
    for chunk in response:
        # Stream only the AIMessageChunk
        if not isinstance(chunk[0], AIMessageChunk):
            continue
        # print (chunk[0].content, chunk[1])
        # Exclude the tool calls that are not part of the conversation
        # if "branch:agent:should_continue:tools" not in chunk[1]["langgraph_triggers"]:
        # if chunk[1]["checkpoint_ns"].startswith("supervisor"):
        #     continue
        if chunk[1]["checkpoint_ns"].startswith("supervisor") is False:
            agent_responding = True
            if "branch:to:agent" in chunk[1]["langgraph_triggers"]:
                if chunk[0].content == "":
                    yield "\n"
                yield chunk[0].content
        else:
            # If no agent has responded yet
            # and the message is from the supervisor
            # then display the message
            if agent_responding is False:
                if "branch:to:agent" in chunk[1]["langgraph_triggers"]:
                    if chunk[0].content == "":
                        yield "\n"
                    yield chunk[0].content
        # if "tools" in chunk[1]["langgraph_triggers"]:
        #     agent_responded = True
        #     if chunk[0].content == "":
        #         yield "\n"
        #     yield chunk[0].content
        # if agent_responding:
        #     continue
        # if "branch:to:agent" in chunk[1]["langgraph_triggers"]:
        #     if chunk[0].content == "":
        #         yield "\n"
        #     yield chunk[0].content


def update_state_t2b(st):
    dic = {
        "sbml_file_path": [st.session_state.sbml_file_path],
        "text_embedding_model": get_text_embedding_model(
            st.session_state.text_embedding_model
        ),
    }
    return dic


def update_state_t2kg(st):
    dic = {
        "embedding_model": get_text_embedding_model(
            st.session_state.text_embedding_model
        ),
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
    }
    return dic


def get_ai_messages(current_state):
    last_msg_is_human = False
    # If only supervisor answered i.e. no agent was called
    if isinstance(current_state.values["messages"][-2], HumanMessage):
        # msgs_to_consider = current_state.values["messages"]
        last_msg_is_human = True
    # else:
    #     # If agent answered i.e. ignore the supervisor msg
    #     msgs_to_consider = current_state.values["messages"][:-1]
    msgs_to_consider = current_state.values["messages"]
    # Get all the AI msgs in the
    # last response from the state
    assistant_content = []
    # print ('LEN:', len(current_state.values["messages"][:-1]))
    # print (current_state.values["messages"][-2])
    # Variable to check if the last message is from the "supervisor"
    # Supervisor message exists for agents that have sub-agents
    # In such cases, the last message is from the supervisor
    # and that is the message to be displayed to the user.
    # for msg in current_state.values["messages"][:-1][::-1]:
    for msg in msgs_to_consider[::-1]:
        if isinstance(msg, HumanMessage):
            break
        if (
            isinstance(msg, AIMessage)
            and msg.content != ""
            and msg.name == "supervisor"
            and last_msg_is_human is False
        ):
            continue
        # Run the following code if the message is from the agent
        if isinstance(msg, AIMessage) and msg.content != "":
            assistant_content.append(msg.content)
            continue
    # Reverse the order
    assistant_content = assistant_content[::-1]
    # Join the messages
    assistant_content = "\n".join(assistant_content)
    return assistant_content


def get_response(agent, graphs_visuals, app, st, prompt):
    # Create config for the agent
    config = {"configurable": {"thread_id": st.session_state.unique_id}}
    # Update the agent state with the selected LLM model
    current_state = app.get_state(config)
    # app.update_state(
    #     config,
    #     {"sbml_file_path": [st.session_state.sbml_file_path]}
    # )
    app.update_state(
        config, {"llm_model": get_base_chat_model(st.session_state.llm_model)}
    )
    # app.update_state(
    #     config,
    #     {"text_embedding_model": get_text_embedding_model(
    #         st.session_state.text_embedding_model),
    #     "embedding_model": get_text_embedding_model(
    #         st.session_state.text_embedding_model),
    #     "uploaded_files": st.session_state.uploaded_files,
    #     "topk_nodes": st.session_state.topk_nodes,
    #     "topk_edges": st.session_state.topk_edges,
    #     "dic_source_graph": [
    #         {
    #             "name": st.session_state.config["kg_name"],
    #             "kg_pyg_path": st.session_state.config["kg_pyg_path"],
    #             "kg_text_path": st.session_state.config["kg_text_path"],
    #         }
    #     ]}
    # )
    if agent == "T2AA4P":
        app.update_state(config, update_state_t2b(st) | update_state_t2kg(st))
    elif agent == "T2B":
        app.update_state(config, update_state_t2b(st))
    elif agent == "T2KG":
        app.update_state(config, update_state_t2kg(st))

    ERROR_FLAG = False
    with collect_runs() as cb:
        # Add Langsmith tracer
        tracer = LangChainTracer(project_name=st.session_state.project_name)
        # Get response from the agent
        if current_state.values["llm_model"]._llm_type == "chat-nvidia-ai-playground":
            response = app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config | {"callbacks": [tracer]},
                # stream_mode="messages"
            )
            # Get the current state of the graph
            current_state = app.get_state(config)
            # Get last response's AI messages
            assistant_content = get_ai_messages(current_state)
            # st.markdown(response["messages"][-1].content)
            st.write(assistant_content)
        else:
            response = app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config | {"callbacks": [tracer]},
                stream_mode="messages",
            )
            st.write_stream(stream_response(response))
        # print (cb.traced_runs)
        # Save the run id and use to save the feedback
        st.session_state.run_id = cb.traced_runs[-1].id

    # Get the current state of the graph
    current_state = app.get_state(config)
    # Get last response's AI messages
    assistant_content = get_ai_messages(current_state)
    # # Get all the AI msgs in the
    # # last response from the state
    # assistant_content = []
    # for msg in current_state.values["messages"][::-1]:
    #     if isinstance(msg, HumanMessage):
    #         break
    #     if isinstance(msg, AIMessage) and msg.content != '':
    #         assistant_content.append(msg.content)
    #         continue
    # # Reverse the order
    # assistant_content = assistant_content[::-1]
    # # Join the messages
    # assistant_content = '\n'.join(assistant_content)
    # Add response to chat history
    assistant_msg = ChatMessage(
        # response["messages"][-1].content,
        # current_state.values["messages"][-1].content,
        assistant_content,
        role="assistant",
    )
    st.session_state.messages.append({"type": "message", "content": assistant_msg})
    # # Display the response in the chat
    # st.markdown(response["messages"][-1].content)
    st.empty()
    # Get the current state of the graph
    current_state = app.get_state(config)
    # Get the messages from the current state
    # and reverse the order
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
            # print ('AIMessage', msg)
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
            msg.name + "_" + msg.tool_call_id + "_" + str(st.session_state.run_id)
        )
        print(uniq_msg_id)
        if msg.name in ["simulate_model", "custom_plotter"]:
            if msg.name == "simulate_model":
                print(
                    "-",
                    len(current_state.values["dic_simulated_data"]),
                    "simulate_model",
                )
                # Convert the simulated data to a single dictionary
                dic_simulated_data = {}
                for data in current_state.values["dic_simulated_data"]:
                    for key in data:
                        if key not in dic_simulated_data:
                            dic_simulated_data[key] = []
                        dic_simulated_data[key] += [data[key]]
                # Create a pandas dataframe from the dictionary
                df_simulated_data = pd.DataFrame.from_dict(dic_simulated_data)
                # Get the simulated data for the current tool call
                df_simulated = pd.DataFrame(
                    df_simulated_data[
                        df_simulated_data["tool_call_id"] == msg.tool_call_id
                    ]["data"].iloc[0]
                )
                df_selected = df_simulated
            elif msg.name == "custom_plotter":
                if msg.artifact:
                    df_selected = pd.DataFrame.from_dict(msg.artifact["dic_data"])
                    # print (df_selected)
                else:
                    continue
            # Display the talbe and plotly chart
            render_table_plotly(
                uniq_msg_id,
                msg.content,
                df_selected,
                x_axis_label=msg.artifact["x_axis_label"],
                y_axis_label=msg.artifact["y_axis_label"],
            )
        elif msg.name == "steady_state":
            if not msg.artifact:
                continue
            # Create a pandas dataframe from the dictionary
            df_selected = pd.DataFrame.from_dict(msg.artifact["dic_data"])
            # Make column 'species_name' the index
            df_selected.set_index("species_name", inplace=True)
            # Display the toggle button to suppress the table
            render_toggle(
                key="toggle_table_" + uniq_msg_id,
                toggle_text="Show Table",
                toggle_state=True,
                save_toggle=True,
            )
            # Display the table
            render_table(df_selected, key="dataframe_" + uniq_msg_id, save_table=True)
        elif msg.name == "search_models":
            if not msg.artifact:
                continue
            # Create a pandas dataframe from the dictionary
            df_selected = pd.DataFrame.from_dict(msg.artifact["dic_data"])
            # Pick selected columns
            df_selected = df_selected[["url", "name", "format", "submissionDate"]]
            # Display the toggle button to suppress the table
            render_toggle(
                key="toggle_table_" + uniq_msg_id,
                toggle_text="Show Table",
                toggle_state=True,
                save_toggle=True,
            )
            # Display the table
            st.dataframe(
                df_selected,
                use_container_width=True,
                key="dataframe_" + uniq_msg_id,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn(
                        label="ID",
                        help="Click to open the link associated with the Id",
                        validate=r"^http://.*$",  # Ensure the link is valid
                        display_text=r"^https://www.ebi.ac.uk/biomodels/(.*?)$",
                    ),
                    "name": st.column_config.TextColumn("Name"),
                    "format": st.column_config.TextColumn("Format"),
                    "submissionDate": st.column_config.TextColumn("Submission Date"),
                },
            )
            # Add data to the chat history
            st.session_state.messages.append(
                {
                    "type": "dataframe",
                    "content": df_selected,
                    "key": "dataframe_" + uniq_msg_id,
                    "tool_name": msg.name,
                }
            )

        elif msg.name == "parameter_scan":
            # Convert the scanned data to a single dictionary
            dic_scanned_data = {}
            for data in current_state.values["dic_scanned_data"]:
                for key in data:
                    if key not in dic_scanned_data:
                        dic_scanned_data[key] = []
                    dic_scanned_data[key] += [data[key]]
            # Create a pandas dataframe from the dictionary
            df_scanned_data = pd.DataFrame.from_dict(dic_scanned_data)
            # Get the scanned data for the current tool call
            df_scanned_current_tool_call = pd.DataFrame(
                df_scanned_data[df_scanned_data["tool_call_id"] == msg.tool_call_id]
            )
            # df_scanned_current_tool_call.drop_duplicates()
            # print (df_scanned_current_tool_call)
            for count in range(0, len(df_scanned_current_tool_call.index)):
                # Get the scanned data for the current tool call
                df_selected = pd.DataFrame(
                    df_scanned_data[
                        df_scanned_data["tool_call_id"] == msg.tool_call_id
                    ]["data"].iloc[count]
                )
                # Display the toggle button to suppress the table
                render_table_plotly(
                    uniq_msg_id + "_" + str(count),
                    df_scanned_current_tool_call["name"].iloc[count],
                    df_selected,
                    x_axis_label=msg.artifact["x_axis_label"],
                    y_axis_label=msg.artifact["y_axis_label"],
                )
        elif msg.name in ["get_annotation"]:
            if not msg.artifact:
                continue
            # Convert the annotated data to a single dictionary
            # print ('-', len(current_state.values["dic_annotations_data"]))
            dic_annotations_data = {}
            for data in current_state.values["dic_annotations_data"]:
                # print (data)
                for key in data:
                    if key not in dic_annotations_data:
                        dic_annotations_data[key] = []
                    dic_annotations_data[key] += [data[key]]
            df_annotations_data = pd.DataFrame.from_dict(dic_annotations_data)
            # Get the annotated data for the current tool call
            df_selected = pd.DataFrame(
                df_annotations_data[
                    df_annotations_data["tool_call_id"] == msg.tool_call_id
                ]["data"].iloc[0]
            )
            # print (df_selected)
            df_selected["Id"] = df_selected.apply(
                lambda row: row["Link"], axis=1  # Ensure "Id" has the correct links
            )
            df_selected = df_selected.drop(columns=["Link"])
            # Directly use the "Link" column for the "Id" column
            render_toggle(
                key="toggle_table_" + uniq_msg_id,
                toggle_text="Show Table",
                toggle_state=True,
                save_toggle=True,
            )
            st.dataframe(
                df_selected,
                use_container_width=True,
                key="dataframe_" + uniq_msg_id,
                hide_index=True,
                column_config={
                    "Id": st.column_config.LinkColumn(
                        label="Id",
                        help="Click to open the link associated with the Id",
                        validate=r"^http://.*$",  # Ensure the link is valid
                        display_text=r"^http://identifiers\.org/(.*?)$",
                    ),
                    "Species Name": st.column_config.TextColumn("Species Name"),
                    "Description": st.column_config.TextColumn("Description"),
                    "Database": st.column_config.TextColumn("Database"),
                },
            )
            # Add data to the chat history
            st.session_state.messages.append(
                {
                    "type": "dataframe",
                    "content": df_selected,
                    "key": "dataframe_" + uniq_msg_id,
                    "tool_name": msg.name,
                }
            )
        elif msg.name in ["subgraph_extraction"]:
            print(
                "-",
                len(current_state.values["dic_extracted_graph"]),
                "subgraph_extraction",
            )
            # Add the graph into the visuals list
            latest_graph = current_state.values["dic_extracted_graph"][-1]
            if current_state.values["dic_extracted_graph"]:
                graphs_visuals.append(
                    {
                        "content": latest_graph["graph_dict"],
                        "key": "subgraph_" + uniq_msg_id,
                    }
                )
        elif msg.name in ["display_dataframe"]:
            # This is a tool of T2S agent's sub-agent S2
            dic_papers = msg.artifact
            if not dic_papers:
                continue
            df_papers = pd.DataFrame.from_dict(dic_papers, orient="index")
            # Add index as a column "key"
            df_papers["Key"] = df_papers.index
            # Drop index
            df_papers.reset_index(drop=True, inplace=True)
            # Drop colum abstract
            # Define the columns to drop
            columns_to_drop = [
                "Abstract",
                "Key",
                "arxiv_id",
                "semantic_scholar_paper_id",
                "source",
                "filename",
                "pdf_url",
                "attachment_key",
            ]

            # Check if columns exist before dropping
            existing_columns = [
                col for col in columns_to_drop if col in df_papers.columns
            ]

            if existing_columns:
                df_papers.drop(columns=existing_columns, inplace=True)

            if "Year" in df_papers.columns:
                df_papers["Year"] = df_papers["Year"].apply(
                    lambda x: (
                        str(int(x)) if pd.notna(x) and str(x).isdigit() else None
                    )
                )

            if "Date" in df_papers.columns:
                df_papers["Date"] = df_papers["Date"].apply(
                    lambda x: (
                        pd.to_datetime(x, errors="coerce").strftime("%Y-%m-%d")
                        if pd.notna(pd.to_datetime(x, errors="coerce"))
                        else None
                    )
                )

            st.dataframe(
                df_papers,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn(
                        display_text="Open",
                    ),
                },
            )
            # Add data to the chat history
            st.session_state.messages.append(
                {
                    "type": "dataframe",
                    "content": df_papers,
                    "key": "dataframe_" + uniq_msg_id,
                    "tool_name": msg.name,
                }
            )
            st.empty()


def render_graph(graph_dict: dict, key: str, save_graph: bool = False):
    """
    Function to render the graph in the chat.

    Args:
        graph_dict: The graph dictionary
        key: The key for the graph
        save_graph: Whether to save the graph in the chat history
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes with attributes
    for node, attrs in graph_dict["nodes"]:
        graph.add_node(node, **attrs)

    # Add edges with attributes
    for source, target, attrs in graph_dict["edges"]:
        graph.add_edge(source, target, **attrs)

    # print("Graph nodes:", graph.nodes(data=True))
    # print("Graph edges:", graph.edges(data=True))

    # Render the graph
    fig = gravis.d3(
        graph,
        node_size_factor=3.0,
        show_edge_label=True,
        edge_label_data_source="label",
        edge_curvature=0.25,
        zoom_factor=1.0,
        many_body_force_strength=-500,
        many_body_force_theta=0.3,
        node_hover_neighborhood=True,
        # layout_algorithm_active=True,
    )
    components.html(fig.to_html(), height=475)

    if save_graph:
        # Add data to the chat history
        st.session_state.messages.append(
            {
                "type": "graph",
                "content": graph_dict,
                "key": key,
            }
        )


def get_text_embedding_model(model_name) -> Embeddings:
    """
    Function to get the text embedding model.

    Args:
        model_name: str: The name of the model

    Returns:
        Embeddings: The text embedding model
    """
    dic_text_embedding_models = {
        "NVIDIA/llama-3.2-nv-embedqa-1b-v2": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "OpenAI/text-embedding-ada-002": "text-embedding-ada-002",
    }
    if model_name.startswith("NVIDIA"):
        return NVIDIAEmbeddings(model=dic_text_embedding_models[model_name])
    return OpenAIEmbeddings(model=dic_text_embedding_models[model_name])


def get_base_chat_model(model_name) -> BaseChatModel:
    """
    Function to get the base chat model.

    Args:
        model_name: str: The name of the model

    Returns:
        BaseChatModel: The base chat model
    """
    dic_llm_models = {
        "NVIDIA/llama-3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
        "NVIDIA/llama-3.1-405b-instruct": "meta/llama-3.1-405b-instruct",
        "NVIDIA/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "OpenAI/gpt-4o-mini": "gpt-4o-mini",
    }
    if model_name.startswith("Llama"):
        return ChatOllama(model=dic_llm_models[model_name], temperature=0)
    elif model_name.startswith("NVIDIA"):
        return ChatNVIDIA(model=dic_llm_models[model_name], temperature=0)
    return ChatOpenAI(model=dic_llm_models[model_name], temperature=0)


@st.dialog("Warning ⚠️")
def update_llm_model():
    """
    Function to update the LLM model.
    """
    llm_model = st.session_state.llm_model
    st.warning(
        f"Clicking 'Continue' will reset all agents, \
            set the selected LLM to {llm_model}. \
            This action will reset the entire app, \
            and agents will lose access to the \
            conversation history. Are you sure \
            you want to proceed?"
    )
    if st.button("Continue"):
        # st.session_state.vote = {"item": item, "reason": reason}
        # st.rerun()
        # Delete all the items in Session state
        for key in st.session_state.keys():
            if key in ["messages", "app"]:
                del st.session_state[key]
        st.rerun()


def update_text_embedding_model(app):
    """
    Function to update the text embedding model.

    Args:
        app: The LangGraph app
    """
    config = {"configurable": {"thread_id": st.session_state.unique_id}}
    app.update_state(
        config,
        {
            "text_embedding_model": get_text_embedding_model(
                st.session_state.text_embedding_model
            )
        },
    )


@st.dialog("Get started with Talk2Biomodels 🚀")
def help_button():
    """
    Function to display the help dialog.
    """
    st.markdown(
        """I am an AI agent designed to assist you with biological
modeling and simulations. I can assist with tasks such as:
1. Search specific models in the BioModels database.

```
Search models on Crohns disease
```

2. Extract information about models, including species, parameters, units,
name and descriptions.

```
Briefly describe model 537 and
its parameters related to drug dosage
```

3. Simulate models:
    - Run simulations of models to see how they behave over time.
    - Set the duration and the interval.
    - Specify which species/parameters you want to include and their starting concentrations/values.
    - Include recurring events.

```
Simulate the model 537 for 2016 hours and
intervals 300 with an initial value
of `DoseQ2W` set to 300 and `Dose` set to 0.
```

4. Answer questions about simulation results.

```
What is the concentration of species IL6 in serum
at the end of simulation?
```

5. Create custom plots to visualize the simulation results.

```
Plot the concentration of all
the interleukins over time.
```

6. Bring a model to a steady state and determine the concentration of a species at the steady state.

```
Bring BioModel 27 to a steady state,
and then determine the Mpp concentration
at the steady state.
```

7. Perform parameter scans to determine the effect of changing parameters on the model behavior.

```
How does the value of Pyruvate change in
model 64 if the concentration of Extracellular Glucose
is changed from 10 to 100 with a step size of 10?
The simulation should run for 5 time units with an
interval of 10.
```

8. Check out the [Use Cases](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2biomodels/cases/Case_1/)
for more examples, and the [FAQs](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2biomodels/faq/)
for common questions.

9. Provide feedback to the developers by clicking on the feedback button.

"""
    )


def apply_css():
    """
    Function to apply custom CSS for streamlit app.
    """
    # Styling using CSS
    st.markdown(
        """<style>
        .stFileUploaderFile { display: none;}
        #stFileUploaderPagination { display: none;}
        .st-emotion-cache-wbtvu4 { display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_file_type_icon(file_type: str) -> str:
    """
    Function to get the icon for the file type.

    Args:
        file_type (str): The file type.

    Returns:
        str: The icon for the file type.
    """
    return {"drug_data": "💊", "multimodal": "📦"}.get(file_type)


@st.fragment
def get_t2b_uploaded_files(app):
    """
    Upload files for T2B agent.
    """
    # Upload the XML/SBML file
    uploaded_sbml_file = st.file_uploader(
        "Upload an XML/SBML file",
        accept_multiple_files=False,
        type=["xml", "sbml"],
        help="Upload a QSP as an XML/SBML file",
    )

    # Upload the article
    article = st.file_uploader(
        "Upload an article",
        help="Upload a PDF article to ask questions.",
        accept_multiple_files=False,
        type=["pdf"],
        key="article",
    )
    # Update the agent state with the uploaded article
    if article:
        # print (article.name)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(article.read())
        # Create config for the agent
        config = {"configurable": {"thread_id": st.session_state.unique_id}}
        # Update the agent state with the selected LLM model
        app.update_state(config, {"pdf_file_name": f.name})
    # Return the uploaded file
    return uploaded_sbml_file


@st.fragment
def initialize_selections() -> None:
    """
    Initialize the selections.

    Args:
        cfg: The configuration object.
    """
    with open(st.session_state.config["kg_pyg_path"], "rb") as f:
        pyg_graph = pickle.load(f)

    # Populate the selections based on the node type from the graph
    selections = {}
    for i in np.unique(np.array(pyg_graph.node_type)):
        selections[i] = []

    return selections


@st.fragment
def get_uploaded_files(cfg: hydra.core.config_store.ConfigStore) -> None:
    """
    Upload files to a directory set in cfg.upload_data_dir, and display them in the UI.

    Args:
        cfg: The configuration object.
    """
    data_package_files = st.file_uploader(
        "💊 Upload pre-clinical drug data",
        help="Free-form text. Must contain atleast drug targets and kinetic parameters",
        accept_multiple_files=True,
        type=cfg.data_package_allowed_file_types,
        key=f"uploader_{st.session_state.data_package_key}",
    )

    multimodal_files = st.file_uploader(
        "📦 Upload multimodal data package",
        help="A spread sheet containing multimodal data package (e.g., genes, drugs, etc.)",
        accept_multiple_files=True,
        type=cfg.multimodal_allowed_file_types,
        key=f"uploader_multimodal_{st.session_state.multimodal_key}",
    )

    # Merge the uploaded files
    uploaded_files = data_package_files.copy()
    if multimodal_files:
        uploaded_files += multimodal_files.copy()

    with st.spinner("Storing uploaded file(s) ..."):
        # for uploaded_file in data_package_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [
                uf["file_name"] for uf in st.session_state.uploaded_files
            ]:
                current_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                uploaded_file.file_name = uploaded_file.name
                uploaded_file.file_path = (
                    f"{cfg.upload_data_dir}/{uploaded_file.file_name}"
                )
                uploaded_file.current_user = st.session_state.current_user
                uploaded_file.timestamp = current_timestamp
                if uploaded_file.name in [uf.name for uf in data_package_files]:
                    uploaded_file.file_type = "drug_data"
                elif uploaded_file.name in [uf.name for uf in multimodal_files]:
                    uploaded_file.file_type = "multimodal"
                st.session_state.uploaded_files.append(
                    {
                        "file_name": uploaded_file.file_name,
                        "file_path": uploaded_file.file_path,
                        "file_type": uploaded_file.file_type,
                        "uploaded_by": uploaded_file.current_user,
                        "uploaded_timestamp": uploaded_file.timestamp,
                    }
                )
                with open(
                    os.path.join(cfg.upload_data_dir, uploaded_file.file_name), "wb"
                ) as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_file = None

    # Display uploaded files and provide a remove button
    for uploaded_file in st.session_state.uploaded_files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(
                get_file_type_icon(uploaded_file["file_type"])
                + uploaded_file["file_name"]
            )
        with col2:
            if st.button("🗑️", key=uploaded_file["file_name"]):
                with st.spinner("Removing uploaded file ..."):
                    if os.path.isfile(
                        f"{cfg.upload_data_dir}/{uploaded_file['file_name']}"
                    ):
                        os.remove(f"{cfg.upload_data_dir}/{uploaded_file['file_name']}")
                    st.session_state.uploaded_files.remove(uploaded_file)
                    st.cache_data.clear()
                    st.session_state.data_package_key += 1
                    st.session_state.multimodal_key += 1
                    st.rerun(scope="fragment")
