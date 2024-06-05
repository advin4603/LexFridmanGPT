from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.core.component import Component
from typing import List, Dict, Any
import csv
import sys
from tqdm import tqdm


def create_indexing_pipeline(document_store: Component, document_embedder: Component = None) -> Pipeline:
    """Create indexing pipeline for encoding documents."""
    if document_embedder is None:
        document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2", meta_fields_to_embed=["title", "guest"])
        document_embedder.warm_up()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(
        split_by="passage", split_length=2, split_overlap=0)

    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline = Pipeline(metadata={})
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline


def run_indexing_pipeline(indexing_pipeline: Pipeline, documents: List[Document]) -> Dict[str, Any]:
    """Pass arguments to the corresponding pipelines and run indexing pipeline"""
    return indexing_pipeline.run({"cleaner": {"documents": documents}})


def load_podcast_csv(filepath: str, progress_bar: bool = True) -> List[Document]:
    """Parse csv file to get a list of podcast documents"""

    # some fields exceed default field length, so extend max field size
    csv.field_size_limit(sys.maxsize)
    if progress_bar:
        with open("podcastdata_dataset.csv") as f:
            lines = sum(1 for _ in f) - 1  # dont count header
    else:
        lines = None

    with open(filepath) as f:
        reader = csv.DictReader(f)
        return [Document(id=episode["id"], content=episode["text"],
                         meta={"guest": episode["guest"], "title": episode["title"]})
                for episode in
                (tqdm(reader, desc="reading", total=lines) if progress_bar else reader)]
