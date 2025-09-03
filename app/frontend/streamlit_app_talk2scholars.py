#!/usr/bin/env python3

"""
Talk2Scholars: A Streamlit app for the Talk2Scholars graph.
"""

import hashlib
import logging
import os
import random
import sys
import tempfile

import hydra
import streamlit as st
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from utils import streamlit_utils

sys.path.append("./")
# import get_app from main_agent


from aiagents4pharma.talk2scholars.agents.main_agent import get_app
from aiagents4pharma.talk2scholars.tools.pdf.utils.generate_answer import (
    load_hydra_config,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore import (
    get_vectorstore,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.paper_loader import load_all_papers
from aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper import (
    ZoteroSearchData,
)

# Set the logging level for Langsmith tracer to ERROR to suppress warnings
logging.getLogger("langsmith").setLevel(logging.ERROR)
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

# Set the logging level for httpx to ERROR to suppress info logs
logging.getLogger("httpx").setLevel(logging.ERROR)
# Initialize configuration

# Initialize configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
if "config" not in st.session_state:
    with hydra.initialize(
        version_base=None,
        config_path="../../aiagents4pharma/talk2scholars/configs",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["app/frontend=default"],
        )
        st.session_state.config = cfg
else:
    cfg = st.session_state.config

# Extract frontend config for backward compatibility
cfg_frontend = cfg.app.frontend

st.set_page_config(
    page_title=cfg_frontend.page.title,
    page_icon=cfg_frontend.page.icon,
    layout=cfg_frontend.page.layout,
    initial_sidebar_state="collapsed",
)


# Set the logo using config
def get_logo_path():
    container_path = cfg_frontend.logo_paths.container
    local_path = cfg_frontend.logo_paths.local

    if os.path.exists(container_path):
        return container_path
    elif os.path.exists(local_path):
        return local_path
    else:
        # Fallback: try to find it relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(script_dir, cfg_frontend.logo_paths.relative)
        if os.path.exists(relative_path):
            return relative_path

    return None  # File not found


logo_path = get_logo_path()
if logo_path:
    st.logo(image=logo_path, size="large", link=cfg_frontend.logo_link)


# Check required environment variables based on config
required_env_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if var not in os.environ]

# Check for optional NVIDIA API key if NVIDIA models are configured
if cfg_frontend.nvidia_llms and "NVIDIA_API_KEY" not in os.environ:
    missing_vars.append("NVIDIA_API_KEY")

if missing_vars:
    st.error(
        f"Please set the {', '.join(missing_vars)} "
        "environment variable(s) in the terminal where you run "
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

# Initialize unified session state
streamlit_utils.initialize_session_state(cfg, agent_type="T2S")

# Initialize the app with default LLM model for the first time
if "app" not in st.session_state:
    # Initialize the app using the utility function for proper model mapping
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


def get_pdf_hash(file_bytes):
    """Generate a SHA-256 hash from PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


@st.fragment
def process_pdf_upload():
    """
    Upload and process multiple PDF files with security validation.
    Saves them as a nested dictionary in session state under 'article_data',
    and updates the LangGraph agent state accordingly.
    """
    # Use secure file upload with validation
    pdf_files = streamlit_utils.secure_file_upload(
        "Upload articles",
        allowed_types=["pdf"],
        help_text="Upload one or more articles in PDF format.",
        max_size_mb=50,  # Reasonable size for academic PDFs
        accept_multiple_files=True,
        key="secure_pdf_upload",
    )

    if pdf_files:
        # Step 1: Initialize or get existing article_data
        article_data = st.session_state.get("article_data", {})

        # Step 2: Process each uploaded file (now pre-validated)
        files_to_process = pdf_files if isinstance(pdf_files, list) else [pdf_files]

        for pdf_file in files_to_process:
            # Sanitize filename for security
            safe_filename = streamlit_utils.sanitize_filename(pdf_file.name)

            file_bytes = pdf_file.read()

            # Generate a stable hash-based ID
            pdf_hash = get_pdf_hash(file_bytes)
            pdf_id = f"uploaded_{pdf_hash}"

            # Prevent duplicates before adding new entry
            if pdf_id in article_data:
                # Optionally skip or update existing
                logging.info(
                    f"Duplicate detected for: {safe_filename}. Skipping re-upload."
                )
                continue

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(file_bytes)
                file_path = f.name

            # Create metadata dict with sanitized filename
            pdf_metadata = {
                "Title": safe_filename,  # Use sanitized filename
                "Authors": ["Uploaded by user"],
                "Abstract": "User uploaded PDF",
                "Publication Date": "N/A",
                "pdf_url": file_path,
                "filename": safe_filename,  # Use sanitized filename
                "source": "upload",
            }

            # Add to the article_data dictionary
            article_data[pdf_id] = pdf_metadata

        # Step 3: Save to session state
        st.session_state.article_data = article_data

        # Step 4: Update LangGraph state
        config: RunnableConfig = {
            "configurable": {"thread_id": st.session_state.unique_id}
        }

        current_state = app.get_state(config)
        if "article_data" not in current_state.values:
            app.update_state(config, {"article_data": {}})

        app.update_state(config, {"article_data": article_data})

        # Final confirmation
        st.success(f"{len(pdf_files)} PDF(s) processed (new or updated).")


def force_collection_reload_after_loading(vector_store, call_id: str = "streamlit"):
    """
    Force reload collection into memory after new papers are loaded.
    This ensures new embeddings are available for fast search.
    """
    logger = logging.getLogger(__name__)

    try:
        # Get the collection from the vector store
        collection = getattr(vector_store.vector_store, "col", None)
        if collection is None:
            collection = getattr(vector_store.vector_store, "collection", None)

        if collection is None:
            logger.warning(f"{call_id}: Cannot access collection for reloading")
            return False

        # Flush to ensure all data is persisted
        logger.info(
            f"{call_id}: Flushing collection to ensure all data is persisted..."
        )
        collection.flush()

        # Get current entity count
        num_entities = collection.num_entities
        hardware_type = "GPU" if vector_store.has_gpu else "CPU"

        logger.info(
            f"{call_id}: Reloading collection with {num_entities} entities into {hardware_type} memory..."
        )

        # Reload collection into memory
        collection.load()

        # Verify the reload
        final_count = collection.num_entities
        logger.info(
            f"{call_id}: Collection successfully reloaded into {hardware_type} memory with {final_count} entities"
        )

        return True

    except Exception as e:
        logger.error(
            f"{call_id}: Failed to reload collection into memory: {e}", exc_info=True
        )
        return False


def initialize_zotero_and_build_store():
    """
    Initializes the Zotero library, downloads PDFs, and builds the Milvus-based vector store.
    Uses a singleton factory pattern to avoid redundant vector store creation.
    Enhanced with proper collection reloading for new embeddings.
    """

    logger = logging.getLogger(__name__)

    try:
        # Initialize Zotero and fetch articles
        app = st.session_state.app

        logger.info("Fetching Zotero articles and downloading PDFs...")
        search_data = ZoteroSearchData(
            query="",
            only_articles=True,
            limit=1,  # get all
            tool_call_id="streamlit_startup",
            download_pdfs=True,
        )
        search_data.process_search()
        article_data = search_data.get_search_results().get("article_data", {})
        st.session_state.article_data = article_data

        logger.info(f"Found {len(article_data)} articles in Zotero library")

        # Update state
        app.update_state(
            {"configurable": {"thread_id": st.session_state.unique_id}},
            {"article_data": article_data},
        )

        # Initialize vector store
        pdf_config = load_hydra_config()
        embedding_model = streamlit_utils.get_text_embedding_model(
            st.session_state.text_embedding_model
        )
        logger.info("Initializing Milvus vector store...")
        vector_store = get_vectorstore(
            embedding_model=embedding_model, config=pdf_config
        )
        st.session_state.vector_store = vector_store

        # Log hardware configuration
        hardware_info = "GPU-accelerated" if vector_store.has_gpu else "CPU-optimized"
        logger.info(f"Vector store initialized in {hardware_info} mode")

        # Prepare papers for loading
        papers_to_load = [
            (paper_id, meta["pdf_url"], meta)
            for paper_id, meta in article_data.items()
            if meta.get("pdf_url")
        ]

        skipped_papers = [
            paper_id
            for paper_id, meta in article_data.items()
            if not meta.get("pdf_url")
        ]

        # Count papers that are actually new (not already loaded)
        papers_already_loaded = len(
            vector_store.loaded_papers.intersection(
                set(paper_id for paper_id, _, _ in papers_to_load)
            )
        )
        papers_to_actually_load = len(papers_to_load) - papers_already_loaded

        logger.info(
            f"Paper status — Total: {len(article_data)}, "
            f"To load (deduped internally): {len(papers_to_load)}, "
            f"Already loaded: {papers_already_loaded}, "
            f"Actually new: {papers_to_actually_load}, "
            f"No PDF: {len(skipped_papers)}"
        )

        if papers_to_load:
            logger.info(f"Starting batch loading of {len(papers_to_load)} papers...")

            # Load papers (this will handle deduplication internally)
            load_all_papers(
                vector_store=vector_store,
                articles=article_data,
                call_id="streamlit_startup",
                config=pdf_config,
                has_gpu=vector_store.has_gpu,
            )

            logger.info("Successfully loaded all papers into vector store.")

            # CRITICAL: Force reload collection if we added new papers OR if it wasn't loaded initially
            # This ensures both new and existing embeddings are in memory
            logger.info(
                "Ensuring collection is properly loaded into memory for fast access..."
            )
            reload_success = force_collection_reload_after_loading(
                vector_store, "streamlit_startup"
            )

            if reload_success:
                logger.info(
                    "Collection successfully loaded into memory - ready for fast searches!"
                )
            else:
                logger.warning("Collection reload failed - searches may be slower")

        else:
            logger.info("All papers are already embedded or skipped.")

            # Even if no new papers, ensure existing collection is loaded
            logger.info("Ensuring existing collection is loaded into memory...")
            force_collection_reload_after_loading(vector_store, "streamlit_startup")

        st.session_state.zotero_initialized = True

        # Log final statistics
        try:
            collection = getattr(vector_store.vector_store, "col", None)
            if collection is not None:
                final_entities = collection.num_entities
                hardware_type = "GPU" if vector_store.has_gpu else "CPU"

                logger.info(
                    f"Zotero initialization complete! "
                    f"{final_entities} document chunks ready in {hardware_type} memory"
                )
            else:
                logger.info("Zotero initialization complete!")
        except Exception as e:
            logger.debug(f"Could not get final entity count: {e}")
            logger.info("Zotero initialization complete!")

    except Exception:
        logger.error(
            "Failed to initialize Zotero and build vector store", exc_info=True
        )
        raise


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

        # LLM panel (Only at the front-end for now)
        llms = tuple(streamlit_utils.get_all_available_llms(cfg))
        st.selectbox(
            "Pick an LLM to power the agent",
            llms,
            index=0,
            key="llm_model",
            on_change=streamlit_utils.update_llm_model,
        )

        # Text embedding model panel
        text_models = tuple(streamlit_utils.get_all_available_embeddings(cfg))
        st.selectbox(
            "Pick a text embedding model",
            text_models,
            index=0,
            key="text_embedding_model",
            on_change=streamlit_utils.update_text_embedding_model,
            kwargs={"app": app},
            help="Used for Retrival Augmented Generation (RAG) and other tasks.",
        )

        # Upload files (placeholder)
        process_pdf_upload()

    with st.container(border=False, height=500):
        prompt = st.chat_input("Say something ...", key="st_chat_input")

# Second column
with main_col2:
    # Chat history panel
    with st.container(border=True, height=775):
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
            elif message["type"] == "button":
                if st.button(message["content"], key=message["key"]):
                    # Trigger the question
                    prompt = message["question"]
                    st.empty()
            elif message["type"] == "dataframe":
                if "tool_name" in message:
                    if message["tool_name"] in [
                        "display_dataframe",
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
        # Display intro message only the first time
        # i.e. when there are no messages in the chat
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner(
                    "Setting up the `agent` and `vector store`. This may take a moment..."
                ):
                    # Initialize Zotero library and RAG index before greeting
                    if not st.session_state.zotero_initialized:
                        initialize_zotero_and_build_store()
                    config: RunnableConfig = {
                        "configurable": {"thread_id": st.session_state.unique_id}
                    }
                    # Prepare LLM and embedding model for updating the agent
                    llm_model = streamlit_utils.get_base_chat_model(
                        st.session_state.llm_model
                    )

                    if cfg_frontend.default_embedding_model == "ollama":
                        emb_model = OllamaEmbeddings(
                            model=cfg_frontend.ollama_embeddings[0]
                        )
                    else:
                        emb_model = OpenAIEmbeddings(
                            model=cfg_frontend.openai_embeddings[0]
                        )

                    # Update the agent state with initial configuration
                    app.update_state(
                        config,
                        {
                            "llm_model": llm_model,
                            "embedding_model": emb_model,
                            "text_embedding_model": streamlit_utils.get_text_embedding_model(
                                st.session_state.text_embedding_model
                            ),
                        },
                    )
                    intro_prompt = "Greet and tell your name and about yourself."
                    intro_prompt += " Also, tell about the agents you can access and ther short description."
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
                        stream_mode="messages",
                    )
                    st.write_stream(streamlit_utils.stream_response(response))
                    current_state = app.get_state(config)
                    # Add response to chat history
                    assistant_msg = ChatMessage(
                        content=current_state.values["messages"][-1].content,
                        role="assistant",
                    )
                    st.session_state.messages.append(
                        {"type": "message", "content": assistant_msg}
                    )
        if len(st.session_state.messages) <= 1:
            for count, question in enumerate(streamlit_utils.sample_questions_t2s()):
                if st.button(
                    f"Q{count + 1}. {question}", key=f"sample_question_{count + 1}"
                ):
                    # Trigger the question
                    prompt = question
                # Add button click to chat history
                st.session_state.messages.append(
                    {
                        "type": "button",
                        "question": question,
                        "content": f"Q{count + 1}. {question}",
                        "key": f"sample_question_{count + 1}",
                    }
                )

        # When the user asks a question
        if prompt:
            # Create a key 'uploaded_file' to read the uploaded file
            # if uploaded_file:
            #     st.session_state.article_pdf = uploaded_file.read().decode("utf-8")

            # Display user prompt
            prompt_msg = ChatMessage(content=prompt, role="user")
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
                            ),
                            "text_embedding_model": streamlit_utils.get_text_embedding_model(
                                st.session_state.text_embedding_model
                            ),
                        },
                    )
                    current_state = app.get_state(config)
                    print("ARTICLE_DATA", len(current_state.values["article_data"]))

                    streamlit_utils.get_response("T2S", None, app, st, prompt)

        # Collect feedback and display the thumbs feedback
        if st.session_state.get("run_id"):
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=_submit_feedback,
                key=f"feedback_{st.session_state.run_id}",
            )
