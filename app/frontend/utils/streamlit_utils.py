#!/usr/bin/env python3

"""
Utils for Streamlit.
"""

import datetime
import hashlib
import magic
import os
import pickle
import re
import tempfile
from typing import Dict, List, Optional, Union

import gravis
import hydra
import networkx as nx
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
from pymilvus import Collection, connections, db


# Security configuration for file uploads
UPLOAD_SECURITY_CONFIG = {
    "max_file_size_mb": 50,  # Maximum file size in MB
    "allowed_extensions": {
        "pdf": ["pdf"],
        "xml": ["xml", "sbml"],
        "spreadsheet": ["xlsx", "xls", "csv"],
        "text": ["txt", "md"],
    },
    "allowed_mime_types": {
        "pdf": ["application/pdf"],
        "xml": ["application/xml", "text/xml", "application/x-xml"],
        "spreadsheet": [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "text/csv",
        ],
        "text": ["text/plain", "text/markdown"],
    },
    "dangerous_extensions": [
        "exe", "bat", "cmd", "com", "pif", "scr", "vbs", "js", "jar",
        "app", "deb", "pkg", "dmg", "rpm", "msi", "dll", "sys", "drv",
        "sh", "bash", "ps1", "py", "pl", "rb", "php", "asp", "jsp"
    ],
    "max_filename_length": 255,
}


class FileUploadError(Exception):
    """Custom exception for file upload validation errors."""
    pass


def validate_uploaded_file(
    uploaded_file,
    allowed_types: List[str],
    max_size_mb: Optional[int] = None
) -> Dict[str, Union[bool, str]]:
    """
    Comprehensive security validation for uploaded files.

    Args:
        uploaded_file: Streamlit uploaded file object
        allowed_types: List of allowed file type categories (e.g., ['pdf', 'xml'])
        max_size_mb: Maximum file size in MB (overrides default if provided)

    Returns:
        Dict with validation results: {'valid': bool, 'error': str, 'warnings': List[str]}

    Raises:
        FileUploadError: If validation fails critically
    """
    if not uploaded_file:
        return {"valid": False, "error": "No file provided", "warnings": []}

    warnings = []
    max_size = (max_size_mb or UPLOAD_SECURITY_CONFIG["max_file_size_mb"]) * 1024 * 1024

    # 1. File name validation
    if len(uploaded_file.name) > UPLOAD_SECURITY_CONFIG["max_filename_length"]:
        return {
            "valid": False,
            "error": f"Filename too long (max {UPLOAD_SECURITY_CONFIG['max_filename_length']} chars)",
            "warnings": warnings
        }

    # Check for dangerous characters in filename
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/', '\0']
    if any(char in uploaded_file.name for char in dangerous_chars):
        return {
            "valid": False,
            "error": "Filename contains dangerous characters",
            "warnings": warnings
        }

    # 2. File extension validation
    file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ""

    if file_ext in UPLOAD_SECURITY_CONFIG["dangerous_extensions"]:
        return {
            "valid": False,
            "error": f"Dangerous file extension '{file_ext}' not allowed",
            "warnings": warnings
        }

    # Check if extension is in allowed types
    allowed_extensions = []
    for file_type in allowed_types:
        if file_type in UPLOAD_SECURITY_CONFIG["allowed_extensions"]:
            allowed_extensions.extend(UPLOAD_SECURITY_CONFIG["allowed_extensions"][file_type])

    if file_ext not in allowed_extensions:
        return {
            "valid": False,
            "error": f"File extension '{file_ext}' not allowed. Allowed: {allowed_extensions}",
            "warnings": warnings
        }

    # 3. File size validation
    file_size = uploaded_file.size
    if file_size > max_size:
        return {
            "valid": False,
            "error": f"File too large ({file_size/1024/1024:.1f}MB). Max: {max_size/1024/1024}MB",
            "warnings": warnings
        }

    if file_size == 0:
        return {"valid": False, "error": "File is empty", "warnings": warnings}

    # 4. MIME type validation (read first bytes to check)
    try:
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer

        # Use python-magic to detect MIME type
        detected_mime = magic.from_buffer(file_content, mime=True)

        # Check if detected MIME type matches allowed types
        allowed_mimes = []
        for file_type in allowed_types:
            if file_type in UPLOAD_SECURITY_CONFIG["allowed_mime_types"]:
                allowed_mimes.extend(UPLOAD_SECURITY_CONFIG["allowed_mime_types"][file_type])

        if detected_mime not in allowed_mimes:
            warnings.append(f"MIME type mismatch: detected '{detected_mime}', expected one of {allowed_mimes}")
            # Don't fail on MIME mismatch for now, just warn

    except Exception as e:
        warnings.append(f"Could not verify MIME type: {str(e)}")

    # 5. Content-based validation
    try:
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer again

        # Check for suspicious content patterns
        # Note: We exclude '<%' from PDFs as it's part of legitimate PDF syntax
        suspicious_patterns = [
            b'<script', b'javascript:', b'vbscript:', b'onload=', b'onerror=',
            b'<?php', b'#!/bin/', b'#!/usr/bin/', b'eval(',
            b'exec(', b'system(', b'shell_exec(', b'passthru(',
        ]

        # Additional patterns that are suspicious only in non-PDF files
        if 'pdf' not in allowed_types:
            suspicious_patterns.extend([b'<%'])  # Only block <% in non-PDF files

        content_lower = file_content.lower()
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                return {
                    "valid": False,
                    "error": f"File contains suspicious content pattern: {pattern.decode('utf-8', errors='ignore')}",
                    "warnings": warnings
                }

        # Additional validation for specific file types
        if 'pdf' in allowed_types and file_ext == 'pdf':
            if not file_content.startswith(b'%PDF-'):
                warnings.append("File extension is PDF but content doesn't match PDF format")
            else:
                # For PDFs, check for truly suspicious patterns (not normal PDF syntax)
                pdf_suspicious_patterns = [
                    b'<script>', b'javascript:', b'vbscript:',
                    b'<?php', b'<% eval', b'<% system', b'<% exec'
                ]
                for pattern in pdf_suspicious_patterns:
                    if pattern in content_lower:
                        return {
                            "valid": False,
                            "error": f"PDF contains suspicious code pattern: {pattern.decode('utf-8', errors='ignore')}",
                            "warnings": warnings
                        }

        elif any(xml_type in allowed_types for xml_type in ['xml']) and file_ext in ['xml', 'sbml']:
            if b'<?xml' not in file_content[:100] and b'<' not in file_content[:10]:
                warnings.append("File extension is XML/SBML but content doesn't appear to be XML")

    except Exception as e:
        warnings.append(f"Content validation error: {str(e)}")

    return {"valid": True, "error": "", "warnings": warnings}


def secure_file_upload(
    label: str,
    allowed_types: List[str],
    help_text: str = "",
    max_size_mb: Optional[int] = None,
    accept_multiple_files: bool = False,
    key: Optional[str] = None
):
    """
    Secure wrapper for st.file_uploader with comprehensive validation.

    Args:
        label: Display label for the file uploader
        allowed_types: List of allowed file type categories
        help_text: Help text to display
        max_size_mb: Maximum file size in MB
        accept_multiple_files: Whether to accept multiple files
        key: Unique key for the uploader widget

    Returns:
        Validated uploaded file(s) or None if validation fails
    """
    # Generate type list for Streamlit
    streamlit_types = []
    for file_type in allowed_types:
        if file_type in UPLOAD_SECURITY_CONFIG["allowed_extensions"]:
            streamlit_types.extend(UPLOAD_SECURITY_CONFIG["allowed_extensions"][file_type])

    # Enhanced help text with security info
    enhanced_help = f"{help_text}\n\n🔒 Security: Max {max_size_mb or UPLOAD_SECURITY_CONFIG['max_file_size_mb']}MB, Types: {streamlit_types}"

    uploaded_files = st.file_uploader(
        label,
        type=streamlit_types,
        help=enhanced_help,
        accept_multiple_files=accept_multiple_files,
        key=key
    )

    if not uploaded_files:
        return None

    # Handle single vs multiple files
    files_to_validate = uploaded_files if accept_multiple_files else [uploaded_files]
    validated_files = []

    for uploaded_file in files_to_validate:
        # Validate the file
        validation_result = validate_uploaded_file(uploaded_file, allowed_types, max_size_mb)

        if not validation_result["valid"]:
            st.error(f"❌ {uploaded_file.name}: {validation_result['error']}")

            # Show helpful tips for the file type
            primary_file_type = allowed_types[0] if allowed_types else "general"
            with st.expander("💡 Upload Tips", expanded=False):
                st.info(get_file_validation_help(primary_file_type))
            continue

        # Show warnings if any
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                st.warning(f"⚠️ {uploaded_file.name}: {warning}")

        # File passed validation
        st.success(f"✅ {uploaded_file.name} validated successfully")
        validated_files.append(uploaded_file)

    # Return single file or list based on accept_multiple_files
    if not validated_files:
        return None

    return validated_files if accept_multiple_files else validated_files[0]


def get_file_validation_help(file_type: str) -> str:
    """
    Get help text for file validation errors.

    Args:
        file_type: The file type that failed validation

    Returns:
        Help text explaining common validation issues
    """
    help_texts = {
        "pdf": """
        📋 PDF Upload Tips:
        • Ensure the file is a legitimate PDF (starts with %PDF-)
        • Some PDF creation tools may embed unexpected content
        • Try re-exporting your PDF from the original source
        • Scanned PDFs are usually safer than text-based PDFs
        """,
        "xml": """
        📋 XML/SBML Upload Tips:
        • Ensure the file starts with <?xml or has XML content
        • Check that the file isn't corrupted
        • SBML files should have proper XML structure
        """,
        "spreadsheet": """
        📋 Spreadsheet Upload Tips:
        • Ensure file is saved in proper Excel/CSV format
        • Avoid files with embedded macros or scripts
        • CSV files should use standard delimiters
        """,
        "general": """
        📋 General Upload Tips:
        • Keep file sizes under the specified limit
        • Use clean, descriptive filenames
        • Avoid files from untrusted sources
        • Contact support if legitimate files are being rejected
        """
    }

    return help_texts.get(file_type, help_texts["general"])


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"|?*\\\/\0]', '_', filename)

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')

    # Prevent directory traversal
    filename = os.path.basename(filename)

    # Ensure filename isn't too long
    max_length = UPLOAD_SECURITY_CONFIG["max_filename_length"]
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    # Ensure it's not empty
    if not filename or filename in ['.', '..']:
        filename = f"uploaded_file_{hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:8]}"

    return filename


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
    toggle_state = st.session_state[f"toggle_plotly_{key.split('plotly_')[1]}"]
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
    toggle_state = st.session_state[f"toggle_table_{key.split('dataframe_')[1]}"]
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


def sample_questions_t2kg():
    """
    Function to get the sample questions for Talk2KnowledgeGraphs.
    """
    questions = [
        'What genes are associated with Crohn\'s disease?',
        "List the drugs that target Interleukin-6 and show their molecular structures",
        "Extract a subgraph for JAK1 and JAK2 genes and visualize their interactions",
        "Find the pathway connections between TNF-alpha and inflammatory bowel disease",
        "What are the drug targets for treating ulcerative colitis?",
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
        "embedding_model": st.session_state.t2kg_emb_model,
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
                lambda row: row["Link"],
                axis=1,  # Ensure "Id" has the correct links
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
                "paper_ids",
                "arxiv_id",
                "pm_id",
                "pmc_id",
                "doi",
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

    def extract_inner_html(html):
        match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
        return match.group(1) if match else html

    figures_inner_html = ""

    for name, subgraph_nodes, subgraph_edges in zip(
        graph_dict["name"], graph_dict["nodes"], graph_dict["edges"], strict=False
    ):
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes with attributes
        for node, attrs in subgraph_nodes:
            graph.add_node(node, **attrs)

        # Add edges with attributes
        for source, target, attrs in subgraph_edges:
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
        # components.html(fig.to_html(), height=475)
        inner_html = extract_inner_html(fig.to_html())
        wrapped_html = f"""
        <div class="graph-content">
            {inner_html}
        </div>
        """

        figures_inner_html += f"""
        <div class="graph-box">
            <h3 class="graph-title">{name}</h3>
            {wrapped_html}
        </div>
        """

    if save_graph:
        # Add data to the chat history
        st.session_state.messages.append(
            {
                "type": "graph",
                "content": graph_dict,
                "key": key,
            }
        )

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            overflow-y: hidden;
            height: 100%;
        }}
        .scroll-container {{
            display: flex;
            overflow-x: auto;
            overflow-y: hidden;
            gap: 1rem;
            padding: 1rem;
            background: #f5f5f5;
            height: 100%;
            box-sizing: border-box;
        }}
        .graph-box {{
            flex: 0 0 auto;
            width: 500px;
            height: 515px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: white;
            padding: 0.5rem;
            box-sizing: border-box;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .graph-title {{
            margin: 0 0 16px 0;  /* Increased bottom margin */
            font-family: Arial, sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center;
        }}
        .graph-content {{
            width: 100%;
            flex-grow: 1;
        }}
        .graph-box svg, .graph-box canvas {{
            max-width: 100% !important;
            max-height: 100% !important;
            height: 100% !important;
            width: 100% !important;
        }}
    </style>
    </head>
    <body>
    <div class="scroll-container">
        {figures_inner_html}
    </div>
    </body>
    </html>
    """
    components.html(full_html, height=550, scrolling=False)


# def render_graph(graph_dict: dict, key: str, save_graph: bool = False):
#     """
#     Function to render the graph in the chat.

#     Args:
#         graph_dict: The graph dictionary
#         key: The key for the graph
#         save_graph: Whether to save the graph in the chat history
#     """
#     # Create a directed graph
#     graph = nx.DiGraph()

#     # Add nodes with attributes
#     for node, attrs in graph_dict["nodes"]:
#         graph.add_node(node, **attrs)

#     # Add edges with attributes
#     for source, target, attrs in graph_dict["edges"]:
#         graph.add_edge(source, target, **attrs)

#     # print("Graph nodes:", graph.nodes(data=True))
#     # print("Graph edges:", graph.edges(data=True))

#     # Render the graph
#     fig = gravis.d3(
#         graph,
#         node_size_factor=3.0,
#         show_edge_label=True,
#         edge_label_data_source="label",
#         edge_curvature=0.25,
#         zoom_factor=1.0,
#         many_body_force_strength=-500,
#         many_body_force_theta=0.3,
#         node_hover_neighborhood=True,
#         # layout_algorithm_active=True,
#     )
#     components.html(fig.to_html(), height=475)

#     if save_graph:
#         # Add data to the chat history
#         st.session_state.messages.append(
#             {
#                 "type": "graph",
#                 "content": graph_dict,
#                 "key": key,
#             }
#         )


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
    return {"article": "📜", "drug_data": "💊", "multimodal": "📦"}.get(file_type)


@st.fragment
def get_t2b_uploaded_files(app):
    """
    Upload files for T2B agent with security validation.
    """
    # Upload the XML/SBML file with security validation
    uploaded_sbml_file = secure_file_upload(
        "Upload an XML/SBML file",
        allowed_types=["xml"],
        help_text="Upload a QSP as an XML/SBML file",
        max_size_mb=25,  # Reasonable size for SBML files
        accept_multiple_files=False,
        key="secure_sbml_upload"
    )

    # Upload the article with security validation
    article = secure_file_upload(
        "Upload an article",
        allowed_types=["pdf"],
        help_text="Upload a PDF article to ask questions.",
        max_size_mb=50,  # PDFs can be larger
        accept_multiple_files=False,
        key=f"secure_article_{st.session_state.t2b_article_key}"
    )

    # Update the agent state with the uploaded article
    if article:
        # Sanitize filename for security
        safe_filename = sanitize_filename(article.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_filename}") as f:
            f.write(article.read())
        # Create config for the agent
        config = {"configurable": {"thread_id": st.session_state.unique_id}}
        # Update the agent state with the selected LLM model
        app.update_state(config, {"pdf_file_name": f.name})

        if article.name not in [
            uf["file_name"] for uf in st.session_state.t2b_uploaded_files
        ]:
            st.session_state.t2b_uploaded_files.append(
                {
                    "file_name": safe_filename,  # Use sanitized filename
                    "file_path": f.name,
                    "file_type": "article",
                    "uploaded_by": st.session_state.current_user,
                    "uploaded_timestamp": datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
            article = None

        # Display the uploaded article
        for uploaded_file in st.session_state.t2b_uploaded_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(
                    get_file_type_icon(uploaded_file["file_type"])
                    + uploaded_file["file_name"]
                )
            with col2:
                if st.button("🗑️", key=uploaded_file["file_path"]):
                    with st.spinner("Removing uploaded file ..."):
                        st.session_state.t2b_uploaded_files.remove(uploaded_file)
                        st.cache_data.clear()
                        st.session_state.t2b_article_key += 1
                        st.rerun(scope="fragment")

    # Return the uploaded file
    return uploaded_sbml_file


@st.fragment
def initialize_selections() -> None:
    """
    Initialize the selections.

    Args:
        cfg: The configuration object.
    """
    # with open(st.session_state.config["kg_pyg_path"], "rb") as f:
    # pyg_graph = pickle.load(f)
    # graph_nodes = pd.read_parquet(st.session_state.config["kg_nodes_path"])
    node_types = st.session_state.config["kg_node_types"]

    # Populate the selections based on the node type from the graph
    selections = {}
    for i in node_types:
        selections[i] = []

    return selections


@st.fragment
def get_uploaded_files(cfg: hydra.core.config_store.ConfigStore) -> None:
    """
    Upload files to a directory set in cfg.upload_data_dir, and display them in the UI.
    Now with comprehensive security validation.

    Args:
        cfg: The configuration object.
    """
    data_package_files = secure_file_upload(
        "💊 Upload pre-clinical drug data",
        allowed_types=["text", "spreadsheet", "pdf"],  # Allow common data formats
        help_text="Free-form text. Must contain atleast drug targets and kinetic parameters",
        max_size_mb=25,
        accept_multiple_files=True,
        key=f"secure_uploader_{st.session_state.data_package_key}",
    )

    multimodal_files = secure_file_upload(
        "📦 Upload multimodal endotype/phenotype data package",
        allowed_types=["spreadsheet"],  # Spreadsheets for structured data
        help_text="A spread sheet containing multimodal endotype/phenotype data package (e.g., genes, drugs, etc.)",
        max_size_mb=50,  # Larger for data packages
        accept_multiple_files=True,
        key=f"secure_uploader_multimodal_{st.session_state.multimodal_key}",
    )

    # Merge the uploaded files (handle None values)
    uploaded_files = []
    if data_package_files:
        uploaded_files = data_package_files if isinstance(data_package_files, list) else [data_package_files]
    if multimodal_files:
        additional_files = multimodal_files if isinstance(multimodal_files, list) else [multimodal_files]
        uploaded_files.extend(additional_files)

    if uploaded_files:
        with st.spinner("Storing uploaded file(s) ..."):
            for uploaded_file in uploaded_files:
                # Sanitize filename for security
                safe_filename = sanitize_filename(uploaded_file.name)

                if safe_filename not in [
                    uf["file_name"] for uf in st.session_state.uploaded_files
                ]:
                    current_timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Determine file type based on which uploader it came from
                    file_type = "drug_data"  # Default
                    if data_package_files and uploaded_file in (data_package_files if isinstance(data_package_files, list) else [data_package_files]):
                        file_type = "drug_data"
                    elif multimodal_files and uploaded_file in (multimodal_files if isinstance(multimodal_files, list) else [multimodal_files]):
                        file_type = "multimodal"

                    safe_file_path = os.path.join(cfg.upload_data_dir, safe_filename)

                    st.session_state.uploaded_files.append(
                        {
                            "file_name": safe_filename,  # Use sanitized filename
                            "file_path": safe_file_path,
                            "file_type": file_type,
                            "uploaded_by": st.session_state.current_user,
                            "uploaded_timestamp": current_timestamp,
                        }
                    )

                    # Write file securely
                    with open(safe_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

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


def setup_milvus(cfg: dict):
    """
    Function to connect to the Milvus database.

    Args:
        cfg: The configuration dictionary containing Milvus connection details.
    """
    # Check if the connection already exists
    if not connections.has_connection(cfg.milvus_db.alias):
        # Create a new connection to Milvus
        # Connect to Milvus
        connections.connect(
            alias=cfg.milvus_db.alias,
            host=cfg.milvus_db.host,
            port=cfg.milvus_db.port,
            user=cfg.milvus_db.user,
            password=cfg.milvus_db.password,
        )
        print("Connected to Milvus database.")
    else:
        print("Already connected to Milvus database.")

    # Use a predefined Milvus database
    db.using_database(cfg.milvus_db.database_name)

    return connections.get_connection_addr(cfg.milvus_db.alias)


def get_cache_edge_index(cfg: dict):
    """
    Function to get the edge index of the knowledge graph in the Milvus collection.
    Due to massive records that we should query to get edge index from the Milvus database,
    we pre-loaded this information when the app is started and stored it in a state.

    Args:
        cfg: The configuration dictionary containing the path to the edge index file.

    Returns:
        The edge index.
    """
    # Load collection
    coll = Collection(f"{cfg.milvus_db.database_name}_edges")
    coll.load()

    batch_size = cfg.milvus_db.query_batch_size
    head_list = []
    tail_list = []
    for start in range(0, coll.num_entities, batch_size):
        end = min(start + batch_size, coll.num_entities)
        print(f"Processing triplet_index range: {start} to {end}")
        batch = coll.query(
            expr=f"triplet_index >= {start} and triplet_index < {end}",
            output_fields=["head_index", "tail_index"],
        )
        head_list.extend([r["head_index"] for r in batch])
        tail_list.extend([r["tail_index"] for r in batch])
    edge_index = [head_list, tail_list]

    # Save the edge index to a file
    with open(cfg.milvus_db.cache_edge_index_path, "wb") as f:
        pickle.dump(edge_index, f)
