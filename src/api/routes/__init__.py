"""
API Layer - Routes
Define the REST API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    ConversationRequest,
    ConversationResponse,
    ErrorResponse,
    MessageResponse,
    SectionRetrievedResponse,
)
from src.application import ConversationGraph, ProcessConversationUseCase
from src.domain import (
    DomainException,
    InvalidMessageException,
    LLMException,
    VectorStoreException,
)

router = APIRouter()


def get_conversation_graph() -> ConversationGraph:
    """
    Dependency to get the conversation graph instance.
    Import here to avoid circular imports.
    """
    from main import get_conversation_graph as _get_graph

    return _get_graph()


@router.post(
    "/conversations/completions",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Process a conversation with RAG",
    description="""
    Process a conversation using RAG (Retrieval-Augmented Generation).
    
    The agent will:
    1. Retrieve relevant information from the vector store
    2. Generate a response based only on the retrieved context
    3. Make up to 2 clarifications if necessary
    4. Escalate to human if more than 2 clarifications are needed
    """,
)
async def process_conversation(
    request: ConversationRequest,
    graph: ConversationGraph = Depends(get_conversation_graph),
) -> ConversationResponse:
    """
    Main endpoint to process conversations with RAG

    Args:
        request: Conversation data (helpdeskId, projectName, messages)
        graph: Injected ConversationGraph instance

    Returns:
        Updated conversation with agent response and retrieved sections
    """
    try:
        # Execute the use case with the shared graph instance
        use_case = ProcessConversationUseCase(conversation_graph=graph)

        conversation = use_case.execute(
            helpdesk_id=request.helpdeskId,
            project_name=request.projectName,
            messages=[msg.model_dump() for msg in request.messages],
        )

        # Convert to response format
        response = ConversationResponse(
            messages=[
                MessageResponse(role=msg.role.value, content=msg.content)
                for msg in conversation.messages
            ],
            messagesHistoryById=[
                MessageResponse(role=msg.role.value, content=msg.content)
                for msg in conversation.message_id_history
            ],
            handoverToHumanNeeded=conversation.handover_to_human_needed,
            sectionsRetrieved=[
                SectionRetrievedResponse(score=section.score, content=section.content)
                for section in conversation.sections_retrieved
            ],
        )

        return response

    except InvalidMessageException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid message: {str(e)}",
        )

    except (VectorStoreException, LLMException) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )

    except DomainException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Domain error: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API is working",
)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
