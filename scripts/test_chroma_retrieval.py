from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from trustworthy_maternal_postpartum_rag.utils.config import get_config

CONFIG_PATH = "configs/pipeline_config.yaml"
CHROMA_PATH = Path("data/chroma_db")

CFG = get_config(CONFIG_PATH)

COLLECTION_NAME = CFG["indexing"].get(
    "collection_name",
    f"{CFG['indexing']['collection_prefix']}_{CFG['run']['version']}"
)

QUERY = "What warning signs after birth mean a mother should seek urgent medical care?"
N_RESULTS = 5

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=CFG["embedding"]["model"],
    device=CFG["embedding"]["device"],
)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
)

print("Collection:", COLLECTION_NAME)
print("Count:", collection.count())

results = collection.query(
    query_texts=[QUERY],
    n_results=N_RESULTS,
    include=["documents", "metadatas", "distances"],
)

for i, (doc, meta, dist) in enumerate(
    zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
    start=1,
):
    print("\n" + "=" * 80)
    print(f"Rank: {i}")
    print(f"Distance: {dist:.4f}")
    print("Publisher:", meta.get("publisher"))
    print("Source tier:", meta.get("source_tier"))
    print("Lifecycle:", meta.get("lifecycle_stage"))
    print("Topic:", meta.get("topic_scope"))
    print("File:", meta.get("source_file"))
    print("Page:", meta.get("page_number"))
    print("\nText preview:")
    print(doc[:700])