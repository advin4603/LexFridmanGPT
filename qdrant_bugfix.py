from typing import List

from haystack import component, Document
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from qdrant_client.http import models


@component
class QdrantRetrieverBugfix(QdrantEmbeddingRetriever):
    """
    Circumvent this issue: https://github.com/deepset-ai/hayhooks/issues/25
    """
    @component.output_types(documents=List[Document])
    def run(
            self,
            query_embedding: List[float],
    ):
        return super(QdrantRetrieverBugfix, self).run(query_embedding=query_embedding)
