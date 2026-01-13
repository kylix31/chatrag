"""
Domain Layer - Models
Defines domain entities following DDD principles
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Chat message roles"""

    USER = "USER"
    AGENT = "AGENT"


class Message(BaseModel):
    """Message Entity - represents a message in the conversation"""

    role: MessageRole
    content: str


class RetrievedSection(BaseModel):
    """Value Object - represents a section retrieved from the vector store"""

    score: float
    content: str


class ConversationState(BaseModel):
    """
    Aggregate Root - represents the complete state of a conversation
    Encapsulates all business logic related to the conversation
    """

    helpdesk_id: int
    project_name: str
    messages: List[Message] = Field(default_factory=list)
    message_id_history: List[Message] = Field(default_factory=list)
    handover_to_human_needed: bool = False
    sections_retrieved: List[RetrievedSection] = Field(default_factory=list)
    clarification_count: int = 0

    def add_user_message(self, content: str) -> None:
        """Adds a user message"""
        self.messages.append(Message(role=MessageRole.USER, content=content))

    def add_agent_message(self, content: str) -> None:
        """Adds an agent message"""
        self.messages.append(Message(role=MessageRole.AGENT, content=content))

    def increment_clarification(self, max_clarifications: int = 2) -> None:
        """
        Increments the clarification counter and marks for handover if limit is exceeded
        Business Rule: Maximum of 2 clarifications per conversation
        """
        self.clarification_count += 1
        if self.clarification_count >= max_clarifications:
            self.handover_to_human_needed = True

    def add_retrieved_sections(self, sections: List[RetrievedSection]) -> None:
        """Adds the sections retrieved from RAG"""
        self.sections_retrieved = sections

    def add_messages_to_history(self, messages: List[dict]) -> None:
        """Adds messages from graph final_state to message_id_history"""
        for msg_dict in messages:
            role = (
                MessageRole.USER
                if msg_dict["role"].upper() == "USER"
                else MessageRole.AGENT
            )
            self.message_id_history.append(
                Message(role=role, content=msg_dict["content"])
            )

    def get_conversation_history(self) -> List[dict]:
        """Returns the conversation history formatted for the LLM"""
        return [
            {"role": msg.role.value.lower(), "content": msg.content}
            for msg in self.messages
        ]
