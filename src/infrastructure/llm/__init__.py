"""
Infrastructure Layer - LLM (OpenAI Chat Model)
Implements the interface with the OpenAI chat model
"""

from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.domain import LLMException
from src.infrastructure.config import get_settings


class OpenAILLM:
    """
    Adapter Pattern - adapts the LangChain OpenAI interface to our application
    Principle: Single Responsibility - responsible only for interacting with the LLM
    """

    SYSTEM_PROMPT = """You are Claudia, a virtual assistant specialized in customer support.

IMPORTANT INSTRUCTIONS:
1. Respond ONLY based on the information provided in the retrieved context
2. If you don't have enough information to respond, ask ONE specific question to clarify
3. Be friendly, professional and concise
4. Use emojis moderately to make the conversation more pleasant
5. NEVER invent information that is not in the context

Your response format should be natural and conversational."""

    def __init__(self):
        """Initializes the OpenAI chat model"""
        settings = get_settings()

        try:
            self.llm = ChatOpenAI(
                api_key=SecretStr(settings.openai_api_key),
                model=settings.openai_chat_model,
                temperature=0.7,
            )
        except Exception as e:
            raise LLMException(f"Error initializing OpenAI LLM: {str(e)}")

    def generate_response(
        self,
        user_message: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        clarification_count: int,
        max_clarifications: int,
    ) -> tuple[str, bool]:
        """
        Generates agent response based on context and history

        Args:
            user_message: Current user message
            context: Context retrieved from vector store
            conversation_history: Conversation history
            clarification_count: Current number of clarifications
            max_clarifications: Maximum number of clarifications allowed

        Returns:
            Tuple with (generated_response, is_clarification)
        """
        try:
            # Prepare the prompt with context
            context_prompt = f"""RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{self._format_history(conversation_history)}

CURRENT USER MESSAGE:
{user_message}

CLARIFICATIONS MADE: {clarification_count}/{max_clarifications}"""

            if clarification_count >= max_clarifications - 1:
                context_prompt += "\n\nWARNING: This is your last chance for clarification. If you need more information after this, inform that the ticket will be escalated."

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=context_prompt),
            ]

            # Generate the response
            response = self.llm.invoke(messages)
            # Ensure response_text is always a string
            response_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

            is_clarification = self._is_clarification(response_text)

            return response_text, is_clarification

        except Exception as e:
            raise LLMException(f"Error generating LLM response: {str(e)}")

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formats the conversation history for the prompt"""
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Agent"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def _is_clarification(self, response: str) -> bool:
        """
        Detects if the response contains a question (clarification)
        Business Rule: Identify when the agent is asking for more information
        """
        return "?" in response
