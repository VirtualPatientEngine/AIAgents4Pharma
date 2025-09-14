#!/usr/bin/env python3

"""
Utils for Streamlit.
"""

import datetime
import os
import re
import tempfile

import gravis
import hydra
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ChatMessage, HumanMessage
from langchain_core.tracers.context import collect_runs
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith import Client

# -----------------------------------------------
# Secure Uploads: Validation and Sanitization
# -----------------------------------------------

# Global upload security configuration
UPLOAD_SECURITY_CONFIG = {
    "max_file_size_mb": 50,  # Global default cap
    "max_filename_length": 255,
    "allowed_extensions": {
        "pdf": ["pdf"],
        "xml": ["xml", "sbml"],
        "spreadsheet": ["xlsx", "xls", "csv"],
        "text": ["txt", "md"],
    },
    "dangerous_extensions": [
        "exe",
        "bat",
        "cmd",
        "com",
        "pif",
        "scr",
        "vbs",
        "js",
        "jar",
        "app",
        "deb",
        "pkg",
        "dmg",
        "rpm",
        "msi",
        "dll",
        "sys",
        "drv",
        "sh",
        "bash",
        "ps1",
        "py",
        "pl",
        "rb",
        "php",
        "asp",
        "jsp",
    ],
}


def _detect_mime(file_name: str, content: bytes | None) -> str | None:
    """Best-effort MIME detection using python-magic if available.

    Falls back to mimetypes based on file extension if libmagic is unavailable.
    """
    try:
        import magic  # type: ignore

        if content is not None:
            m = magic.Magic(mime=True)
            return m.from_buffer(content[:4096])  # use header bytes
    except Exception:
        pass
    import mimetypes

    mime, _ = mimetypes.guess_type(file_name)
    return mime


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent traversal and unsafe chars.

    - Remove path components
    - Replace unsafe chars with underscore
    - Enforce max length
    """
    # Strip directory components
    base = os.path.basename(filename)
    # Remove Windows drive letters and colons
    base = re.sub(r"^[A-Za-z]:\\", "", base)
    # Replace dangerous characters
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    # Collapse repeated underscores
    base = re.sub(r"_+", "_", base).strip("._-")
    # Enforce max length
    max_len = UPLOAD_SECURITY_CONFIG["max_filename_length"]
    if len(base) > max_len:
        name, ext = os.path.splitext(base)
        base = name[: max_len - len(ext)] + ext
    # Avoid empty name
    return base or "file"


def validate_uploaded_file(
    uploaded_file, allowed_types: list[str], max_size_mb: int | None = None
) -> dict:
    """Validate a file against security policy.

    Returns a dict with keys: valid: bool, error: str|None, warnings: list[str]
    """
    result = {"valid": True, "error": None, "warnings": []}

    # Derive extension
    original_name = getattr(uploaded_file, "name", "")
    ext = original_name.split(".")[-1].lower() if "." in original_name else ""

    # Block obviously dangerous extensions regardless of allowlist
    if ext in UPLOAD_SECURITY_CONFIG["dangerous_extensions"]:
        result["valid"] = False
        result["error"] = f"File extension '{ext}' is not allowed."
        return result

    # Build master allowed extension list from categories
    allowed_exts = set()
    for t in allowed_types:
        allowed_exts.update(UPLOAD_SECURITY_CONFIG["allowed_extensions"].get(t, []))

    if ext not in allowed_exts:
        result["valid"] = False
        result["error"] = (
            f"File extension '{ext}' not allowed. Allowed: {sorted(allowed_exts)}"
        )
        return result

    # Size check
    max_size = (max_size_mb or UPLOAD_SECURITY_CONFIG["max_file_size_mb"]) * 1024 * 1024
    size_bytes = getattr(uploaded_file, "size", None)
    if size_bytes is None:
        try:
            pos = uploaded_file.tell()
            uploaded_file.seek(0, os.SEEK_END)
            size_bytes = uploaded_file.tell()
            uploaded_file.seek(pos)
        except Exception:
            size_bytes = 0
    if size_bytes and size_bytes > max_size:
        result["valid"] = False
        mb = size_bytes / (1024 * 1024)
        result["error"] = (
            f"File too large ({mb:.1f}MB). Max: {max_size_mb or UPLOAD_SECURITY_CONFIG['max_file_size_mb']}MB"
        )
        return result

    # Read small header/body for scanning and MIME detection
    content = None
    try:
        pos = uploaded_file.tell()
        content = uploaded_file.read(min(size_bytes or (512 * 1024), 512 * 1024))
        uploaded_file.seek(pos)
    except Exception:
        content = None

    # MIME verification
    mime = _detect_mime(original_name, content)
    expected_mimes_by_type = {
        "pdf": {"application/pdf"},
        "xml": {"application/xml", "text/xml"},
        "spreadsheet": {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "text/csv",
            "application/csv",
        },
        "text": {"text/plain", "text/markdown"},
    }
    expected_mimes = set()
    for t in allowed_types:
        expected_mimes |= expected_mimes_by_type.get(t, set())
    # Only warn if we couldn't reliably detect
    if mime and expected_mimes and mime not in expected_mimes:
        result["warnings"].append(
            f"MIME type mismatch: detected '{mime}', expected one of {sorted(expected_mimes)}"
        )

    # Content scanning for dangerous patterns
    if content:
        lower = content[:65536].decode(errors="ignore").lower()
        # Common dangerous patterns
        dangerous_patterns = [
            r"<script",
            r"javascript:",
            r"vbscript:",
            r"<\?php",
            r"#!/bin/",
            r"shell_exec\(",
            r"\beval\(",
            r"\bexec\(",
            r"\bsystem\(",
        ]
        # Additional template/server code marker, but avoid false positives for PDFs
        if "pdf" not in allowed_types:
            dangerous_patterns.append(r"<%")
        else:
            # PDFs: allow <% generally but block explicit code injections
            dangerous_patterns += [r"<%\s*eval", r"<%\s*system"]

        for pat in dangerous_patterns:
            if re.search(pat, lower):
                result["valid"] = False
                result["error"] = f"File contains suspicious content pattern: {pat}"
                return result

        # Light-weight header validation
        if "pdf" in allowed_types and not lower.lstrip().startswith("%pdf-"):
            result["warnings"].append("Missing expected PDF header (%PDF-)")
        if "xml" in allowed_types and "<?xml" not in lower[:200]:
            result["warnings"].append("Missing expected XML header (<?xml)")

    return result


def secure_file_upload(
    label: str,
    allowed_types: list[str],
    help_text: str | None = None,
    max_size_mb: int | None = None,
    accept_multiple_files: bool = False,
    key: str | None = None,
    override_extensions: list[str] | None = None,
):
    """Wrapper around st.file_uploader with validation and sanitization.

    Returns a single UploadedFile, a list of UploadedFile, or None.
    Adds attribute 'sanitized_name' to returned file objects for safe saving.
    """
    # Build extension whitelist for Streamlit uploader
    if override_extensions is not None:
        ext_whitelist = override_extensions
    else:
        ext_whitelist = []
        for t in allowed_types:
            ext_whitelist += UPLOAD_SECURITY_CONFIG["allowed_extensions"].get(t, [])

    files = st.file_uploader(
        label,
        help=help_text,
        accept_multiple_files=accept_multiple_files,
        type=ext_whitelist if ext_whitelist else None,
        key=key,
    )

    if not files:
        return None

    files_list = files if accept_multiple_files else [files]
    accepted = []

    for f in files_list:
        result = validate_uploaded_file(f, allowed_types, max_size_mb)
        if not result["valid"]:
            st.error(f"❌ {getattr(f, 'name', 'file')}: {result['error']}")
            continue
        for w in result["warnings"]:
            st.warning(f"⚠️ {getattr(f, 'name', 'file')}: {w}")
        # Attach sanitized name for downstream use
        try:
            f.sanitized_name = sanitize_filename(getattr(f, "name", "file"))
        except Exception:
            pass
        accepted.append(f)

    if not accepted:
        return None
    if accept_multiple_files:
        return accepted
    return accepted[0]


def resolve_logo(cfg) -> str | None:
    """
    Resolve a logo path from config with safe fallbacks.

    Args:
        cfg: Hydra configuration object with app.frontend.logo_paths

    Returns:
        str | None: Path to logo image if found, else None
    """
    try:
        container_path = cfg.app.frontend.logo_paths.container
        local_path = cfg.app.frontend.logo_paths.local
        relative_cfg = cfg.app.frontend.logo_paths.relative
    except Exception:
        # Minimal fallback if config paths are missing
        container_path = "/app/docs/assets/VPE.png"
        local_path = "docs/assets/VPE.png"
        relative_cfg = "../../docs/assets/VPE.png"

    if os.path.exists(container_path):
        return container_path
    if os.path.exists(local_path):
        return local_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # __file__ is utils dir; adjust to app dir for relative
    # Go up one to frontend
    script_dir = os.path.dirname(script_dir)
    relative_path = os.path.join(script_dir, relative_cfg)
    if os.path.exists(relative_path):
        return relative_path
    return None


def get_azure_token_provider():
    """
    Get Azure AD token provider for Azure OpenAI authentication.

    Returns:
        token provider for Azure AD authentication
    """
    try:
        return get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
    except Exception as e:
        st.error(f"Failed to create Azure token provider: {e}")
        return None


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
        "What genes are associated with Crohn's disease?",
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
            st.session_state.text_embedding_model, st.session_state.config
        ),
    }
    # If a PDF has been uploaded in this session, include it every turn
    pdf_path = st.session_state.get("pdf_file_path")
    if pdf_path:
        dic["pdf_file_name"] = pdf_path
    return dic


def update_state_t2kg(st):
    # Get the config from session state
    cfg = st.session_state.config

    # For T2AA4P, database config is stored separately in t2kg_config
    # For T2KG, it's in the main config
    if hasattr(st.session_state, "t2kg_config"):
        # T2AA4P case - use the separate T2KG config for database info
        db_config = st.session_state.t2kg_config
        database_name = db_config.utils.database.milvus.milvus_db.database_name
    else:
        # T2KG case - use main config
        database_name = cfg.utils.database.milvus.milvus_db.database_name

    dic = {
        "embedding_model": st.session_state.t2kg_emb_model,
        "uploaded_files": st.session_state.uploaded_files,
        "topk_nodes": st.session_state.topk_nodes,
        "topk_edges": st.session_state.topk_edges,
        "dic_source_graph": [
            {
                "name": database_name,
            }
        ],
        "selections": st.session_state.selections,
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
        config,
        {
            "llm_model": get_base_chat_model(
                st.session_state.llm_model, st.session_state.config
            )
        },
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


def get_text_embedding_model(model_name, cfg=None) -> Embeddings:
    """
    Function to get the text embedding model.

    Args:
        model_name: str: The name of the model
        cfg: Optional[DictConfig]: Configuration object containing retry/timeout settings

    Returns:
        Embeddings: The text embedding model
    """
    # Get retry and timeout settings from config or use defaults
    max_retries = 3  # Default for embeddings
    timeout = 30  # Default for embeddings

    if cfg and hasattr(cfg, "app") and hasattr(cfg.app, "frontend"):
        max_retries = getattr(cfg.app.frontend, "embedding_max_retries", 3)
        timeout = getattr(cfg.app.frontend, "embedding_timeout", 30)
    dic_text_embedding_models = {
        "NVIDIA/llama-3.2-nv-embedqa-1b-v2": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "OpenAI/text-embedding-ada-002": "text-embedding-ada-002",
        "Azure/text-embedding-ada-002": "text-embedding-ada-002",
        "nomic-embed-text": "nomic-embed-text",
    }

    if model_name.startswith("NVIDIA"):
        return NVIDIAEmbeddings(model=dic_text_embedding_models[model_name])
    elif model_name.startswith("Azure/"):
        # Azure OpenAI Embeddings configuration
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not azure_endpoint or not azure_deployment:
            st.error(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT environment variables"
            )
            return OpenAIEmbeddings(
                model=dic_text_embedding_models[model_name],
                max_retries=max_retries,
                timeout=timeout,
            )  # Fallback to regular OpenAI

        # Get Azure token provider
        token_provider = get_azure_token_provider()
        if not token_provider:
            st.error("Failed to get Azure token provider")
            return OpenAIEmbeddings(
                model=dic_text_embedding_models[model_name],
                max_retries=max_retries,
                timeout=timeout,
            )  # Fallback to regular OpenAI

        from langchain_openai.embeddings import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
            max_retries=max_retries,
            timeout=timeout,
        )
    elif model_name in dic_text_embedding_models and not model_name.startswith(
        ("OpenAI/", "NVIDIA/", "Azure/")
    ):
        # Ollama embeddings (models without provider prefix)
        return OllamaEmbeddings(model=dic_text_embedding_models[model_name])
    else:
        # Default to OpenAI
        model_key = (
            model_name if model_name.startswith("OpenAI/") else f"OpenAI/{model_name}"
        )
        return OpenAIEmbeddings(
            model=dic_text_embedding_models.get(
                model_key, model_name.replace("OpenAI/", "")
            ),
            max_retries=max_retries,
            timeout=timeout,
        )


def get_base_chat_model(model_name, cfg=None) -> BaseChatModel:
    """
    Function to get the base chat model.

    Args:
        model_name: str: The name of the model
        cfg: Optional[DictConfig]: Configuration object containing retry/timeout settings

    Returns:
        BaseChatModel: The base chat model
    """
    # Get retry and timeout settings from config or use defaults
    max_retries = 5  # Default
    timeout = 60  # Default

    if cfg and hasattr(cfg, "app") and hasattr(cfg.app, "frontend"):
        max_retries = getattr(cfg.app.frontend, "llm_max_retries", 5)
        timeout = getattr(cfg.app.frontend, "llm_timeout", 60)
    dic_llm_models = {
        "NVIDIA/llama-3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
        "NVIDIA/llama-3.1-405b-instruct": "meta/llama-3.1-405b-instruct",
        "NVIDIA/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "OpenAI/gpt-4o-mini": "gpt-4o-mini",
        "Azure/gpt-4o-mini": "gpt-4o-mini",  # Azure model mapping
        "Ollama/llama3.1:8b": "llama3.1:8b",  # Ollama model mapping
    }

    if model_name.startswith("Llama"):
        return ChatOllama(
            model=dic_llm_models[model_name], temperature=0, timeout=timeout
        )
    elif model_name.startswith("Ollama/"):
        return ChatOllama(
            model=dic_llm_models[model_name], temperature=0, timeout=timeout
        )
    elif model_name.startswith("NVIDIA"):
        return ChatNVIDIA(
            model=dic_llm_models[model_name], temperature=0, timeout=timeout
        )
    elif model_name.startswith("Azure/"):
        # Azure OpenAI configuration
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        model_name_env = os.environ.get(
            "AZURE_OPENAI_MODEL_NAME", dic_llm_models[model_name]
        )
        model_version = os.environ.get("AZURE_OPENAI_MODEL_VERSION")

        if not azure_endpoint or not azure_deployment:
            st.error(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT environment variables"
            )
            return ChatOpenAI(
                model=dic_llm_models[model_name],
                temperature=0,
                max_retries=max_retries,
                timeout=timeout,
            )  # Fallback to regular OpenAI

        # Get Azure token provider
        token_provider = get_azure_token_provider()
        if not token_provider:
            st.error("Failed to get Azure token provider")
            return ChatOpenAI(
                model=dic_llm_models[model_name],
                temperature=0,
                max_retries=max_retries,
                timeout=timeout,
            )  # Fallback to regular OpenAI

        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            model_name=model_name_env,
            model_version=model_version,
            azure_ad_token_provider=token_provider,
            temperature=0,
            max_retries=max_retries,
            timeout=timeout,
        )
    elif model_name.startswith("OpenAI/"):
        # Regular OpenAI with optional custom base URL
        base_url = os.environ.get("OPENAI_BASE_URL")
        return ChatOpenAI(
            model=dic_llm_models[model_name],
            temperature=0,
            base_url=base_url if base_url else None,
            max_retries=max_retries,
            timeout=timeout,
        )

    # Default fallback
    return ChatOpenAI(
        model=dic_llm_models.get(model_name, model_name),
        temperature=0,
        max_retries=max_retries,
        timeout=timeout,
    )


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
                st.session_state.text_embedding_model, st.session_state.config
            )
        },
    )


def update_t2kg_embedding_model():
    """
    Update the T2KG embedding model in session from the selected text embedding model.
    """
    st.session_state.t2kg_emb_model = get_text_embedding_model(
        st.session_state.text_embedding_model, st.session_state.config
    )


@st.dialog("How to use this application 🚀")
def help_button():
    """
    Function to display the help dialog.
    """
    st.markdown(
        """**FAQs**
- Talk2KnowledgeGraphs: [click here](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2knowledgegraphs/faq/)
- Talk2Biomodels: [click here](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2biomodels/faq/)

**Video**
- [Watch the app overview](https://www.youtube.com/watch?v=3cU_OxY4HiE)

[![Watch the app overview](https://img.youtube.com/vi/3cU_OxY4HiE/0.jpg)](https://www.youtube.com/watch?v=3cU_OxY4HiE)
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
    Upload files for T2B agent with secure validation.
    """
    # Upload the XML/SBML file securely
    uploaded_sbml_file = secure_file_upload(
        "Upload an XML/SBML file",
        allowed_types=["xml"],
        help_text="Upload a QSP as an XML/SBML file",
        max_size_mb=25,
        accept_multiple_files=False,
        key="secure_sbml_upload",
    )

    # Upload the article securely
    article = secure_file_upload(
        "Upload an article",
        allowed_types=["pdf"],
        help_text="Upload a PDF article to ask questions.",
        max_size_mb=50,
        accept_multiple_files=False,
        key=f"secure_article_upload_{st.session_state.t2b_article_key}",
    )

    # Update the agent state with the uploaded article
    if article:
        # print (article.name)
        safe_name = getattr(article, "sanitized_name", sanitize_filename(article.name))
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_name}") as f:
            f.write(article.read())
        # Create config for the agent
        config = {"configurable": {"thread_id": st.session_state.unique_id}}
        # Update the agent state with the PDF file name and text embedding model
        app.update_state(
            config,
            {
                "pdf_file_name": f.name,
                "text_embedding_model": get_text_embedding_model(
                    st.session_state.text_embedding_model, st.session_state.config
                ),
            },
        )

        # Persist PDF path in session for subsequent turns
        st.session_state.pdf_file_path = f.name

        display_name = safe_name
        if display_name not in [
            uf["file_name"] for uf in st.session_state.t2b_uploaded_files
        ]:
            st.session_state.t2b_uploaded_files.append(
                {
                    "file_name": display_name,
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
                        # Clear PDF reference if this was the active one
                        if (
                            st.session_state.get("pdf_file_path")
                            and st.session_state.pdf_file_path
                            == uploaded_file["file_path"]
                        ):
                            del st.session_state.pdf_file_path
                        st.cache_data.clear()
                        st.session_state.t2b_article_key += 1
                        st.rerun(scope="fragment")

    # Return the uploaded file
    return uploaded_sbml_file


@st.fragment
def initialize_selections() -> dict:
    """
    Initialize the selections based on configured node types.

    Returns:
        dict: Dictionary of node types with empty lists for selections
    """
    try:
        # Load configuration from the session state
        cfg = st.session_state.config

        # For T2AA4P, database config is stored separately in t2kg_config
        # For T2KG, it's in the main config
        if hasattr(st.session_state, "t2kg_config"):
            # T2AA4P case - use the separate T2KG config for database info
            db_config = st.session_state.t2kg_config
            if hasattr(db_config.utils.database.milvus, "kg_node_types"):
                node_types = db_config.utils.database.milvus.kg_node_types
            else:
                node_types = None
        else:
            # T2KG case - use main config
            if hasattr(cfg, "utils") and hasattr(
                cfg.utils.database.milvus, "kg_node_types"
            ):
                node_types = cfg.utils.database.milvus.kg_node_types
            else:
                node_types = None

        # If no node types found in config, use fallback
        if node_types is None:
            # Fallback to default node types from PrimeKG
            node_types = [
                "anatomy",
                "biological_process",
                "cellular_component",
                "compound",
                "disease",
                "drug",
                "effect_phenotype",
                "gene_protein",
                "molecular_function",
                "pathway",
                "side_effect",
            ]

        # Populate the selections based on the node type from the configuration
        selections = {}
        for node_type in node_types:
            selections[node_type] = []

        return selections

    except Exception as e:
        st.error(f"Failed to initialize selections: {str(e)}")
        # Return empty selections as fallback
        return {}


@st.fragment
def get_uploaded_files(cfg: hydra.core.config_store.ConfigStore) -> None:
    """
    Upload files to a directory set in cfg.upload_data_dir, and display them in the UI.

    Args:
        cfg: The configuration object.
    """

    def _exts_to_categories(exts: list[str]) -> list[str]:
        categories = set()
        for e in exts:
            e = e.lower()
            for cat, cat_exts in UPLOAD_SECURITY_CONFIG["allowed_extensions"].items():
                if e in cat_exts:
                    categories.add(cat)
        return list(categories) if categories else []

    data_exts = cfg.app.frontend.data_package_allowed_file_types
    data_categories = _exts_to_categories(data_exts)
    data_package_files = secure_file_upload(
        "💊 Upload pre-clinical drug data",
        allowed_types=data_categories or ["text", "spreadsheet", "pdf"],
        help_text="Free-form text. Must contain atleast drug targets and kinetic parameters",
        max_size_mb=25,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.data_package_key}",
        override_extensions=data_exts,
    )

    multimodal_exts = cfg.app.frontend.multimodal_allowed_file_types
    multimodal_categories = _exts_to_categories(multimodal_exts)
    multimodal_files = secure_file_upload(
        "📦 Upload multimodal endotype/phenotype data package",
        allowed_types=multimodal_categories or ["spreadsheet"],
        help_text="A spread sheet containing multimodal endotype/phenotype data package (e.g., genes, drugs, etc.)",
        max_size_mb=50,
        accept_multiple_files=True,
        key=f"uploader_multimodal_{st.session_state.multimodal_key}",
        override_extensions=multimodal_exts,
    )

    # Merge the uploaded files
    uploaded_files = []
    if data_package_files:
        uploaded_files += (
            data_package_files
            if isinstance(data_package_files, list)
            else [data_package_files]
        )
    if multimodal_files:
        uploaded_files += (
            multimodal_files
            if isinstance(multimodal_files, list)
            else [multimodal_files]
        )

    with st.spinner("Storing uploaded file(s) ..."):
        # for uploaded_file in data_package_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [
                uf["file_name"] for uf in st.session_state.uploaded_files
            ]:
                current_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                safe_name = getattr(
                    uploaded_file,
                    "sanitized_name",
                    sanitize_filename(uploaded_file.name),
                )
                uploaded_file.file_name = safe_name
                uploaded_file.file_path = (
                    f"{cfg.app.frontend.upload_data_dir}/{uploaded_file.file_name}"
                )
                uploaded_file.current_user = st.session_state.current_user
                uploaded_file.timestamp = current_timestamp
                # Determine file_type by source list membership when lists are present
                try:
                    if data_package_files and uploaded_file in (
                        data_package_files
                        if isinstance(data_package_files, list)
                        else [data_package_files]
                    ):
                        uploaded_file.file_type = "drug_data"
                    elif multimodal_files and uploaded_file in (
                        multimodal_files
                        if isinstance(multimodal_files, list)
                        else [multimodal_files]
                    ):
                        uploaded_file.file_type = "multimodal"
                    else:
                        uploaded_file.file_type = "drug_data"
                except Exception:
                    uploaded_file.file_type = "drug_data"
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
                    os.path.join(
                        cfg.app.frontend.upload_data_dir, uploaded_file.file_name
                    ),
                    "wb",
                ) as f:
                    # Ensure buffer is read from start
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
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
                        f"{cfg.app.frontend.upload_data_dir}/{uploaded_file['file_name']}"
                    ):
                        os.remove(
                            f"{cfg.app.frontend.upload_data_dir}/{uploaded_file['file_name']}"
                        )
                    st.session_state.uploaded_files.remove(uploaded_file)
                    st.cache_data.clear()
                    st.session_state.data_package_key += 1
                    st.session_state.multimodal_key += 1
                    st.rerun(scope="fragment")


def get_all_available_llms(cfg):
    """
    Get all available LLM models from configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        list: List of all available LLM model names
    """
    azure_llms = cfg.app.frontend.get("azure_openai_llms", [])
    ollama_llms = cfg.app.frontend.get("ollama_llms", [])

    all_llms = (
        cfg.app.frontend.get("openai_llms", [])
        + cfg.app.frontend.get("nvidia_llms", [])
        + azure_llms
        + ollama_llms
    )

    return all_llms


def get_all_available_embeddings(cfg):
    """
    Get all available embedding models from configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        list: List of all available embedding model names
    """
    azure_embeddings = cfg.app.frontend.get("azure_openai_embeddings", [])
    ollama_embeddings = cfg.app.frontend.get("ollama_embeddings", [])

    all_embeddings = (
        cfg.app.frontend.get("openai_embeddings", [])
        + cfg.app.frontend.get("nvidia_embeddings", [])
        + azure_embeddings
        + ollama_embeddings
    )

    return all_embeddings


def initialize_session_state(cfg, agent_type="T2B"):
    """
    Initialize unified session state for all AI Agents 4 Pharma apps.

    Args:
        cfg: Hydra configuration object
        agent_type: str: Type of agent ("T2B", "T2KG", "T2S", "T2AA4P")
    """
    import os
    import random

    import streamlit as st

    # Core configuration
    if "config" not in st.session_state:
        st.session_state.config = cfg

    if "current_user" not in st.session_state:
        st.session_state.current_user = cfg.app.frontend.default_user

    # Chat and messaging
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "project_name" not in st.session_state:
        st.session_state.project_name = f"{agent_type}-" + str(
            random.randint(1000, 9999)
        )

    if "run_id" not in st.session_state:
        st.session_state.run_id = None

    if "unique_id" not in st.session_state:
        st.session_state.unique_id = random.randint(1, 1000)

    # File management (common across all apps)
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        # Make upload directory if not exists
        upload_dir = cfg.app.frontend.get("upload_data_dir", "../files")
        os.makedirs(upload_dir, exist_ok=True)

    # Model configuration
    if "llm_model" not in st.session_state:
        all_llms = get_all_available_llms(cfg)
        st.session_state.llm_model = all_llms[0] if all_llms else "OpenAI/gpt-4o-mini"

    if "text_embedding_model" not in st.session_state:
        all_embeddings = get_all_available_embeddings(cfg)
        # Default to OpenAI text-embedding-ada-002 unless config overrides
        default_embedding = "OpenAI/text-embedding-ada-002"
        if default_embedding in all_embeddings:
            st.session_state.text_embedding_model = default_embedding
        else:
            st.session_state.text_embedding_model = (
                all_embeddings[0] if all_embeddings else "OpenAI/text-embedding-ada-002"
            )

    # Agent-specific initializations
    if agent_type == "T2KG":
        # Knowledge graph specific session state
        if "selections" not in st.session_state:
            st.session_state.selections = initialize_selections()

        if "data_package_key" not in st.session_state:
            st.session_state.data_package_key = 0

        if "multimodal_key" not in st.session_state:
            st.session_state.multimodal_key = 0

        if "topk_nodes" not in st.session_state:
            st.session_state.topk_nodes = cfg.app.frontend.get(
                "reasoning_subgraph_topk_nodes", 15
            )

        if "topk_edges" not in st.session_state:
            st.session_state.topk_edges = cfg.app.frontend.get(
                "reasoning_subgraph_topk_edges", 15
            )

        # Ensure T2KG has an embedding model in session
        if "t2kg_emb_model" not in st.session_state:
            if cfg.app.frontend.get("default_embedding_model", "openai") == "ollama":
                from langchain_ollama import OllamaEmbeddings

                st.session_state.t2kg_emb_model = OllamaEmbeddings(
                    model=cfg.app.frontend.get(
                        "ollama_embeddings", ["nomic-embed-text"]
                    )[0]
                )
            else:
                from langchain_openai import OpenAIEmbeddings

                st.session_state.t2kg_emb_model = OpenAIEmbeddings(
                    model=cfg.app.frontend.get(
                        "openai_embeddings", ["text-embedding-ada-002"]
                    )[0]
                )

    elif agent_type == "T2B":
        # Biomodels specific session state
        if "sbml_file_path" not in st.session_state:
            st.session_state.sbml_file_path = None

        # Keys used by the T2B upload fragment (PDF/article handling)
        if "t2b_article_key" not in st.session_state:
            st.session_state.t2b_article_key = 0

        if "t2b_uploaded_files" not in st.session_state:
            st.session_state.t2b_uploaded_files = []

    elif agent_type == "T2S":
        # Scholars specific session state
        if "article_data" not in st.session_state:
            st.session_state.article_data = {}

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None

        if "zotero_initialized" not in st.session_state:
            st.session_state.zotero_initialized = False

    elif agent_type == "T2AA4P":
        # Combined agent specific session state (hybrid of T2B + T2KG)

        # T2B specific session state
        if "sbml_file_path" not in st.session_state:
            st.session_state.sbml_file_path = None

        # T2B article upload key
        if "t2b_article_key" not in st.session_state:
            st.session_state.t2b_article_key = 0

        # T2B uploaded files (separate from T2KG files)
        if "t2b_uploaded_files" not in st.session_state:
            st.session_state.t2b_uploaded_files = []

        # T2KG specific session state
        if "selections" not in st.session_state:
            st.session_state.selections = initialize_selections()

        if "data_package_key" not in st.session_state:
            st.session_state.data_package_key = 0

        if "multimodal_key" not in st.session_state:
            st.session_state.multimodal_key = 0

        # Special for T2AA4P: patient gene expression data
        if "endotype_key" not in st.session_state:
            st.session_state.endotype_key = 0

        if "topk_nodes" not in st.session_state:
            st.session_state.topk_nodes = cfg.app.frontend.get(
                "reasoning_subgraph_topk_nodes", 15
            )

        if "topk_edges" not in st.session_state:
            st.session_state.topk_edges = cfg.app.frontend.get(
                "reasoning_subgraph_topk_edges", 15
            )

        # T2KG embedding model (special handling for T2AA4P)
        if "t2kg_emb_model" not in st.session_state:
            if cfg.app.frontend.get("default_embedding_model", "openai") == "ollama":
                from langchain_ollama import OllamaEmbeddings

                st.session_state.t2kg_emb_model = OllamaEmbeddings(
                    model=cfg.app.frontend.get(
                        "ollama_embeddings", ["nomic-embed-text"]
                    )[0]
                )
            else:
                from langchain_openai import OpenAIEmbeddings

                st.session_state.t2kg_emb_model = OpenAIEmbeddings(
                    model=cfg.app.frontend.get(
                        "openai_embeddings", ["text-embedding-ada-002"]
                    )[0]
                )
