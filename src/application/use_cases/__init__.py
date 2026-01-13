"""
Application Layer - Use Cases
Implements the application's use cases
"""

from src.application.graph import ConversationGraph
from src.domain import ConversationState, InvalidMessageException, MessageRole


class ProcessConversationUseCase:
    """
    Use Case: Process a conversation
    Pattern: Use Case / Application Service
    Principle: Single Responsibility - orchestrates business logic
    """

    def __init__(self, conversation_graph: ConversationGraph):
        """Initializes the use case with necessary dependencies"""
        self.conversation_graph = conversation_graph

    def execute(
        self, helpdesk_id: int, project_name: str, messages: list[dict]
    ) -> ConversationState:
        """
        Executes the use case to process a conversation

        Args:
            helpdesk_id: Helpdesk ID
            project_name: Project name
            messages: List of conversation messages

        Returns:
            Updated conversation state with agent response

        Raises:
            InvalidMessageException: If messages are invalid
        """
        if not messages or len(messages) == 0:
            raise InvalidMessageException(
                "The conversation must have at least one message"
            )

        last_message = messages[-1]
        if last_message.get("role") != "USER":
            raise InvalidMessageException("The last message must be from the user")

        conversation = self._build_conversation_state(
            helpdesk_id=helpdesk_id, project_name=project_name, messages=messages
        )

        updated_conversation = self.conversation_graph.process_conversation(
            conversation
        )

        return updated_conversation

    def _build_conversation_state(
        self, helpdesk_id: int, project_name: str, messages: list[dict]
    ) -> ConversationState:
        """
        Builds the conversation state from input data
        """
        conversation = ConversationState(
            helpdesk_id=helpdesk_id, project_name=project_name
        )

        # Rebuild message history
        for msg in messages:
            role = MessageRole(msg["role"])
            content = msg["content"]

            if role == MessageRole.USER:
                conversation.add_user_message(content)
            else:
                conversation.add_agent_message(content)

        return conversation
