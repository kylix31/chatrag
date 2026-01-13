"""
API Layer - Schemas (DTOs)
Defines the API input and output schemas using Pydantic
"""

from typing import List

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    """DTO for input message"""

    role: str = Field(..., pattern="^(USER|AGENT)$")
    content: str = Field(..., min_length=1)


class ConversationRequest(BaseModel):
    """DTO for conversation request"""

    helpdeskId: int = Field(..., alias="helpdeskId", gt=0)
    projectName: str = Field(..., alias="projectName", min_length=1)
    messages: List[MessageRequest] = Field(..., min_length=1)

    class Config:
        populate_by_name = True


class MessageResponse(BaseModel):
    """DTO for response message"""

    role: str
    content: str


class SectionRetrievedResponse(BaseModel):
    """DTO for retrieved section"""

    score: float
    content: str


class ConversationResponse(BaseModel):
    """DTO for conversation response"""

    messages: List[MessageResponse]
    handoverToHumanNeeded: bool = Field(..., alias="handoverToHumanNeeded")
    sectionsRetrieved: List[SectionRetrievedResponse] = Field(
        ..., alias="sectionsRetrieved"
    )

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    """DTO for error response"""

    detail: str
