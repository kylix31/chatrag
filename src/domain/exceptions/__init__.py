"""
Domain Layer - Exceptions
Defines custom domain exceptions
"""


class DomainException(Exception):
    """Base exception for domain errors"""

    pass


class VectorStoreException(DomainException):
    """Exception related to vector store"""

    pass


class LLMException(DomainException):
    """Exception related to LLM"""

    pass


class InvalidMessageException(DomainException):
    """Exception for invalid messages"""

    pass


class MaxClarificationsExceededException(DomainException):
    """Exception when clarification limit is exceeded"""

    pass
