from haystack.components.others import Multiplexer
from haystack.core.component import Component
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret

from documents_output import DocumentsOutput
from hyde_component import HyDE
from gemini_bugfix import GeminiBugfix


def create_rag_pipeline(retriever: Component,
                        text_embedder: Component = None,
                        generator: Component = None,
                        prompt_template: str = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    Document: {{ document.id }}
    Guest: {{ document.meta.guest }}
    Title: {{ document.meta.title }}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""") -> Pipeline:
    """
    Create the RAG Pipeline. Defaults to using Gemini with Qdrant document store.
    """
    if text_embedder is None:
        text_embedder = HyDE(n_completions=1)

    if generator is None:
        generator = GeminiBugfix(model='gemini-1.5-flash', api_key=Secret.from_env_var("GOOGLE_API_KEY"))

    prompt_builder = PromptBuilder(template=prompt_template, required_variables=["question"])
    rag_pipeline = Pipeline(metadata={})

    rag_pipeline.add_component("input", Multiplexer(str))

    # Add components to your pipeline
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)
    rag_pipeline.add_component("documents_output", DocumentsOutput())

    # Now, connect the components to each other
    rag_pipeline.connect("input.value", "prompt_builder.question")
    rag_pipeline.connect("input.value", "text_embedder.query")
    rag_pipeline.connect("text_embedder.hypothetical_embedding",
                         "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("retriever", "documents_output.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    return rag_pipeline


def run_rag_pipeline(rag_pipeline: Pipeline, question: str):
    """Pass arguments to the pipeline components and run it."""
    return rag_pipeline.run(
        {"input": {"value": question}})
