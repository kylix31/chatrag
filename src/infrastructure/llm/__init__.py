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

    SYSTEM_PROMPT = """Você é Claudia, uma assistente virtual especializada em suporte ao cliente.

INSTRUÇÕES IMPORTANTES:
1. Responda APENAS com base nas informações fornecidas no contexto recuperado
2. Se não tiver informações suficientes para responder, faça UMA pergunta específica para esclarecer
3. Seja amigável, profissional e concisa
4. Use emojis de forma moderada para tornar a conversa mais agradável
5. NUNCA invente informações que não estejam no contexto

Formato da sua resposta deve ser natural e conversacional."""

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
            context_prompt = f"""CONTEXTO RECUPERADO:
{context}

HISTÓRICO DA CONVERSA:
{self._format_history(conversation_history)}

MENSAGEM ATUAL DO USUÁRIO:
{user_message}

CLARIFICAÇÕES FEITAS: {clarification_count}/{max_clarifications}"""

            if clarification_count >= max_clarifications - 1:
                context_prompt += "\n\nAVISO: Esta é sua última chance de clarificação. Se precisar de mais informações após esta, informe que o ticket será escalado."

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
            return "Nenhuma conversa anterior."

        formatted = []
        for msg in history:
            role = "Usuário" if msg["role"] == "user" else "Agente"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def _is_clarification(self, response: str) -> bool:
        """
        Detects if the response contains a question (clarification)
        Business Rule: Identify when the agent is asking for more information
        """
        return "?" in response
