import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.config import config
from state.shared_state import shared_state

# Load environment variables from .env file
load_dotenv()


# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. Please check your .env file."
    )


def create_llm() -> ChatOpenAI:
    """Create and configure OpenAI LLM instance"""
    return ChatOpenAI(
        model="gpt-4o-mini",  # Using gpt-4o-mini as requested
        temperature=config.TEMPERATURE,
        timeout=60,  # Timeout in seconds
        max_retries=3,
        api_key=OPENAI_API_KEY,  # Explicitly passing API key
        top_p=0.95,  # Moved out of model_kwargs
        presence_penalty=0,  # Moved out of model_kwargs
        frequency_penalty=0,  # Moved out of model_kwargs
    )


class LLMManager:
    def __init__(self):
        # Initialize the LLM with default configuration
        self.llm = create_llm()

    def get_response(
        self,
        system_prompt: str,
        user_input: str,
        additional_context: Dict[str, Any] = None,
        include_history: bool = True,
    ) -> str:
        """Get response from LLM with system prompt and user input"""
        try:
            # Create messages list
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]

            # Add chat history if requested
            if include_history:
                history = shared_state.get_chat_history(limit=3)
                for msg in history:
                    if msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))

            # Add debug logging
            print("\nDebug - LLM Input:")
            print(f"System prompt: {system_prompt[:200]}...")
            print(f"User input: {user_input}")

            # Get response with retries
            response = self.llm.invoke(messages)

            # Add debug logging
            print("\nDebug - LLM Response:")
            print(f"Raw response: {response.content}")

            # Log token usage if available
            if hasattr(response, "usage_metadata"):
                print(f"Token usage: {response.usage_metadata}")

            if response and response.content.strip():
                return response.content.strip()

            return ""

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return ""

    def bind_tools(self, tools: List[Any], strict: bool = True) -> None:
        """Bind tools to the LLM for function/tool calling with strict mode enabled"""
        self.llm = self.llm.bind_tools(
            tools,
            tool_choice="auto",  # Let the model decide which tool to use
            strict=strict,  # Enable strict mode for better schema validation
        )


# Create a global instance
llm_manager = LLMManager()
