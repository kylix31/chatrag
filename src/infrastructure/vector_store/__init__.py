"""
Infrastructure Layer - Vector Store (Azure AI Search)
Implements the interface with Azure AI Search for document retrieval
"""

from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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

            # Initialize Azure Search client
            credential = AzureKeyCredential(settings.azure_search_key)
            self.search_client = SearchClient(
                endpoint=settings.azure_search_endpoint,
                index_name=settings.azure_search_index_name,
                credential=credential,
            )

        except Exception as e:
            raise VectorStoreException(f"Error initializing Azure AI Search: {str(e)}")

    def similarity_search(
        self, query: str, k: int = 5, project_name: str | None = None
    ) -> List[RetrievedSection]:
        """
        Performs similarity search in the vector store with optional project filter

        Args:
            query: User query
            k: Number of documents to return
            project_name: Optional project name to filter results (e.g., 'tesla_motors')

        Returns:
            List of retrieved sections with score
        """
        try:
            # Generate embeddings for the query
            query_vector = self.embeddings.embed_query(query)

            # Build filter expression for Azure AI Search
            filter_expression = None
            if project_name:
                filter_expression = f"projectName eq '{project_name}'"

            # Build vector query using VectorizedQuery
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields="embeddings",
            )

            # Perform vector search using Azure Search SDK
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filter_expression,
                select=["content", "type"],
                top=k,
            )

            # Convert to domain format
            sections = []
            for result in results:
                # Azure Cognitive Search returns @search.score
                score = result.get("@search.score", 0.0)
                content = result.get("content", "")
                sections.append(RetrievedSection(score=score, content=content))

            return sections

        except Exception as e:
            raise VectorStoreException(f"Error in vector search: {str(e)}")
