"""
Domain Layer - Models
Define as entidades de domínio seguindo princípios de DDD
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles das mensagens no chat"""

    USER = "USER"
    AGENT = "AGENT"


class Message(BaseModel):
    """Entidade Message - representa uma mensagem na conversa"""

    role: MessageRole
    content: str


class RetrievedSection(BaseModel):
    """Value Object - representa uma seção recuperada do vector store"""

    score: float
    content: str


class ConversationState(BaseModel):
    """
    Aggregate Root - representa o estado completo de uma conversa
    Encapsula toda a lógica de negócio relacionada à conversa
    """

    helpdesk_id: int
    project_name: str
    messages: List[Message] = Field(default_factory=list)
    handover_to_human_needed: bool = False
    sections_retrieved: List[RetrievedSection] = Field(default_factory=list)
    clarification_count: int = 0

    def add_user_message(self, content: str) -> None:
        """Adiciona uma mensagem do usuário"""
        self.messages.append(Message(role=MessageRole.USER, content=content))

    def add_agent_message(self, content: str) -> None:
        """Adiciona uma mensagem do agente"""
        self.messages.append(Message(role=MessageRole.AGENT, content=content))

    def increment_clarification(self, max_clarifications: int = 2) -> None:
        """
        Incrementa o contador de clarificações e marca para handover se exceder o limite
        Business Rule: Máximo de 2 clarificações por conversa
        """
        self.clarification_count += 1
        if self.clarification_count >= max_clarifications:
            self.handover_to_human_needed = True

    def add_retrieved_sections(self, sections: List[RetrievedSection]) -> None:
        """Adiciona as seções recuperadas do RAG"""
        self.sections_retrieved = sections

    def get_conversation_history(self) -> List[dict]:
        """Retorna o histórico de conversa formatado para o LLM"""
        return [
            {"role": msg.role.value.lower(), "content": msg.content}
            for msg in self.messages
        ]
