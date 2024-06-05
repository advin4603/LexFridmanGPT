from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from dotenv import load_dotenv
from pprint import pprint
from haystack.utils import Secret
from os import environ

from qdrant_bugfix import QdrantRetrieverBugfix
import document_indexing
import rag

load_dotenv()

document_store = QdrantDocumentStore(
    environ.get("QDRANT_URL"), index="Document", embedding_dim=384, recreate_index=False,
    api_key=Secret.from_env_var("QDRANT_API_KEY"), https=True, timeout=20, metadata={})
retriever = QdrantRetrieverBugfix(document_store)

indexing_pipeline = document_indexing.create_indexing_pipeline(document_store)
with open("pipelines/indexing_pipeline.yaml", "w") as f:
    indexing_pipeline.dump(f)
document_indexing.run_indexing_pipeline(indexing_pipeline,
                                        document_indexing.load_podcast_csv("podcastdata_dataset.csv"))

rag_pipeline = rag.create_rag_pipeline(retriever)

with open("pipelines/rag_pipeline.yaml", "w") as f:
    rag_pipeline.dump(f)

question = "What does Lex Fridman discuss about the ethical implications of AI?"

response = rag.run_rag_pipeline(rag_pipeline, question)

pprint(response)
