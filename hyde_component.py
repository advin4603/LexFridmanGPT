from haystack import component, Document, Pipeline, default_to_dict, default_from_dict
from haystack.core.component import Component
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from typing import List, Dict, Any
from gemini_bugfix import GeminiBugfix
import numpy as np


def document_filter(answers):
    return [Document(content=a) for a in answers]


@component
class HyDE:
    """Component that implements Hypothetical Document Embeddings.
    Instead of encoding query, generate an hypothetical document using the query and use that to retrieve documents for RAG"""

    def __init__(self,
                 generator: Component = None,
                 n_completions: int = 5,
                 temperature: float = 0.75,
                 max_tokens: int = 400,
                 instruction: str = "Given a question, generate a paragraph of text that answers the question.",
                 input_label: str = "Question",
                 embedder: Component = None
                 ):
        if generator is None:
            self.generator = GeminiBugfix(model='gemini-1.5-flash', generation_config={
                "candidate_count": n_completions, "temperature": temperature, "max_output_tokens": max_tokens})

        self.n_completions = n_completions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.input_label = input_label

        self.prompt_builder = PromptBuilder(
            template=f"{instruction}\n{input_label}: ""{{ query }}\nParagraph: ")

        self.adapter = OutputAdapter(
            template="{{answers | document_create}}",
            output_type=List[Document],
            custom_filters={"document_create": document_filter}
        )

        if embedder is None:
            self.embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2")
            self.embedder.warm_up()
        else:
            self.embedder = embedder

        self.pipeline = Pipeline(metadata={})
        self.pipeline.add_component(
            name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(
            name="generator", instance=self.generator)
        self.pipeline.add_component(name="adapter", instance=self.adapter)
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.answers")
        self.pipeline.connect("adapter.output", "embedder.documents")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(self, generator=None,
                               n_completions=5,
                               temperature=0.75,
                               max_tokens=400,
                               instruction="Given a question, generate a paragraph of text that answers the question.",
                               input_label="Question",
                               embedder=None)
        data["pipeline"] = self.pipeline.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyDE":
        hyde_obj = default_from_dict(cls, data)
        hyde_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyde_obj

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, query: str):
        result = self.pipeline.run(data={"prompt_builder": {"query": query}})

        hyde_vector = np.array(
            [doc.embedding for doc in result["embedder"]["documents"]]
        ).mean(axis=0).reshape((1, -1))

        return {"hypothetical_embedding": hyde_vector[0].tolist()}
