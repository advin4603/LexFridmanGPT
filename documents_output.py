from dataclasses import dataclass
from typing import Dict, Any, List
from haystack import component, Document


@component
class DocumentsOutput:
    """
    A component to adapt the output of retriever to circumvent this issue: https://github.com/deepset-ai/hayhooks/issues/25
    """

    @component.output_types(documents=List[Dict[str, Any]])
    def run(self, documents: List[Document]):
        return {"documents": [dict(
            id=document.id,
            content=document.content,
            meta=document.meta,
            score=document.score
        ) for document in documents]}
