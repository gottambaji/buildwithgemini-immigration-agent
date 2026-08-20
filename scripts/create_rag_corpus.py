import os
import vertexai
from vertexai.preview import rag
from vertexai.preview.rag.utils import resources as rr

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-03-76f4257706a9")
LOCATION = "us-central1"  # Serverless RAG Engine is supported in us-central1
GCS_PATH = f"gs://{PROJECT_ID}-visa-docs/rag/"

PARSING_PROMPT = (
    "Extract useful non-immigrant legal guidance, 221(g) administrative processing procedures, "
    "and lawyer consultation instructions. Output clean, self-contained prose."
)

print(f"Initializing Vertex AI RAG Engine for project '{PROJECT_ID}' in location '{LOCATION}'...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 1. Switch region's RAG managed DB to serverless mode
cfg = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragEngineConfig"
rag.update_rag_engine_config(
    rag_engine_config=rag.RagEngineConfig(
        name=cfg,
        rag_managed_db_config=rag.RagManagedDbConfig(mode=rr.Serverless()),
    )
)

# 2. Create corpus
print("Creating RAG Engine corpus 'visa-legal-corpus'...")
corpus = rag.create_corpus(
    display_name="visa-legal-corpus",
    embedding_model_config=rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-005"
    ),
)
print(f"✅ RAG Corpus Created: {corpus.name}")

# 3. Import & Chunk files with LLM parser
print(f"Importing files from '{GCS_PATH}' into RAG corpus...")
resp = rag.import_files(
    corpus_name=corpus.name,
    paths=[GCS_PATH],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
    ),
    llm_parser=rag.LlmParserConfig(
        model_name="gemini-2.5-flash",
        custom_parsing_prompt=PARSING_PROMPT
    ),
)
print(f"✅ Successfully imported {resp.imported_rag_files_count} file(s) into RAG Corpus!")
