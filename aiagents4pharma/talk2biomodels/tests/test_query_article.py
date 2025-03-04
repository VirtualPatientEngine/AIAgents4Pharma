'''
Test cases for Talk2Biomodels query_article tool.
'''

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from ..agents.t2b_agent import get_app

LLM_MODEL = ChatOpenAI(model='gpt-4o-mini', temperature=0)

class Article(BaseModel):
    '''
    Article schema.
    '''
    title: str = Field(description="Title of the article.")

def test_query_article_with_an_article():
    '''
    Test the query_article tool by providing an article.
    '''
    unique_id = 12345
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    # Update state by providing the pdf file name
    # and the text embedding model
    app.update_state(config,
      {"pdf_file_name": "aiagents4pharma/talk2biomodels/tests/article_on_model_537.pdf",
       "text_embedding_model": NVIDIAEmbeddings(model='nvidia/llama-3.2-nv-embedqa-1b-v2')})
    prompt = "What is the title of the article?"
    # Test the tool query_article
    response = app.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
    # Get the response from the tool
    assistant_msg = response["messages"][-1].content
    # Prepare a LLM that can be used as a judge
    llm = LLM_MODEL
    # Make it return a structured output
    structured_llm = llm.with_structured_output(Article)
    # Prepare a prompt for the judge
    prompt = "Given the text below, what is the title of the article?"
    prompt += f"\n\n{assistant_msg}"
    # Get the structured output
    article = structured_llm.invoke(prompt)
    # Check if the article title is correct
    expected_title = "A Multiscale Model of IL-6–Mediated "
    expected_title += "Immune Regulation in Crohn’s Disease"
    # Check if the article title is correct
    assert article.title == expected_title

def test_query_article_without_an_article():
    '''
    Test the query_article tool without providing an article.
    The status of the tool should be error.
    '''
    unique_id = 12345
    app = get_app(unique_id, llm_model=LLM_MODEL)
    config = {"configurable": {"thread_id": unique_id}}
    prompt = "What is the title of the uploaded article?"
    # Update state by providing the text embedding model
    app.update_state(config,
      {"text_embedding_model": NVIDIAEmbeddings(model='nvidia/llama-3.2-nv-embedqa-1b-v2')})
    # Test the tool query_article
    app.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )
    current_state = app.get_state(config)
    # Get the messages from the current state
    # and reverse the order
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages
    # until a ToolMessage is found.
    tool_status_is_error = False
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            # Skip until it finds a ToolMessage
            if msg.name == "query_article" and msg.status == "error":
                tool_status_is_error = True
                break
    assert tool_status_is_error
