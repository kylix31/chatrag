"""
Infrastructure Layer - Vector Store (Azure AI Search)
Implements the interface with Azure AI Search for document retrieval
"""

from typing import List

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from src.domain import RetrievedSection, VectorStoreException
from src.infrastructure.config import get_settings


class AzureAISearchVectorStore:
    """
    Repository Pattern - encapsulates Azure AI Search access
    Principle: Dependency Inversion - depends on abstractions (interfaces) not concrete implementations
    """

    def __init__(self):
        """Initializes the connection with Azure AI Search"""
        settings = get_settings()

        try:
            self.embeddings = OpenAIEmbeddings(
                api_key=SecretStr(settings.openai_api_key),
                model=settings.openai_embedding_model,
            )

            self.vector_store = AzureSearch(
                azure_search_endpoint=settings.azure_search_endpoint,
                azure_search_key=settings.azure_search_key,
                index_name=settings.azure_search_index_name,
                embedding_function=self.embeddings.embed_query,
            )

        except Exception as e:
            raise VectorStoreException(f"Error initializing Azure AI Search: {str(e)}")

    def similarity_search(self, query: str, k: int = 5) -> List[RetrievedSection]:
        """
        Performs similarity search in the vector store

        Args:
            query: User query
            k: Number of documents to return

        Returns:
            List of retrieved sections with score
        """
        try:
            # Performs the search using LangChain retriever
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query, k=k
            )

            # Converts to domain format
            sections = [
                RetrievedSection(score=score, content=doc.page_content)
                for doc, score in results
            ]

            return sections

        except Exception as e:
            raise VectorStoreException(f"Error in vector search: {str(e)}")
