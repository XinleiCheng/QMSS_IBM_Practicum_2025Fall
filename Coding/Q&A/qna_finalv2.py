# Import packages
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Runtime and performance settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Environment setup
load_dotenv()

# Basic config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

db_root_config = os.getenv(
    "CHROMA_ROOT",
    os.path.join("Data", "Vector DataBase"),
)
DB_ROOT = (
    db_root_config
    if os.path.isabs(db_root_config)
    else os.path.join(REPOSITORY_ROOT, db_root_config)
)

TOP_K = int(os.getenv("TOP_K", "6"))
SEM_THRESHOLD = float(os.getenv("SEM_THRESHOLD", "0.75"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5",
)
QUERY_INSTRUCTION = os.getenv(
    "QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

DB_EPLC_PATH = os.path.join(DB_ROOT, "chroma_db_EPLC final Phase")
DB_HHS_PATH = os.path.join(DB_ROOT, "chroma_eplc_policy")


def require_existing_database(path: str, label: str) -> None:
    database_file = os.path.join(path, "chroma.sqlite3")
    if not os.path.isfile(database_file):
        raise FileNotFoundError(
            f"{label} database not found at {path}. "
            "Set CHROMA_ROOT to the directory containing the Q&A databases."
        )


require_existing_database(DB_EPLC_PATH, "EPLC")
require_existing_database(DB_HHS_PATH, "HHS")

# Initialize embedding model
sbert = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

# Connect to DBs
eplc_db = PersistentClient(path=DB_EPLC_PATH)
hhs_db  = PersistentClient(path=DB_HHS_PATH)

def get_single_collection(db, label):
    cols = db.list_collections()
    if not cols:
        raise RuntimeError(f"[error] No collections in {label}")
    if len(cols) > 1:
        raise RuntimeError(f"[error] Multiple collections in {label}")
    return cols[0]

coll_eplc = get_single_collection(eplc_db, "EPLC")
coll_hhs = get_single_collection(hhs_db, "HHS")


# ---------------- Retrieval ----------------------

def query_collection(collection, database_label, query_embedding, k):
    response = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    distances = response.get("distances", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]

    results = []
    for document_id, document, distance, metadata in zip(
        ids,
        documents,
        distances,
        metadatas,
    ):
        if not document:
            continue
        results.append(
            {
                "id": document_id,
                "document": document,
                "distance": float(distance),
                "metadata": metadata or {},
                "database": database_label,
            }
        )
    return results


def deduplicate_results(results):
    best_result_by_document = {}
    for result in results:
        document_key = " ".join(result["document"].split()).lower()
        existing_result = best_result_by_document.get(document_key)
        if (
            existing_result is None
            or result["distance"] < existing_result["distance"]
        ):
            best_result_by_document[document_key] = result
    return list(best_result_by_document.values())


def retrieve(query: str, k: int = TOP_K):
    query = query.strip()
    if not query:
        return []

    instructed_query = f"{QUERY_INSTRUCTION}{query}"
    query_embedding = sbert.encode(
        [instructed_query],
        normalize_embeddings=True,
    ).tolist()

    results = query_collection(
        coll_eplc,
        "EPLC",
        query_embedding,
        k,
    )
    results.extend(
        query_collection(
            coll_hhs,
            "HHS",
            query_embedding,
            k,
        )
    )

    results = deduplicate_results(results)
    results.sort(key=lambda result: result["distance"])
    return results[:k]


def filter_relevant_results(results, threshold=SEM_THRESHOLD):
    return [
        result
        for result in results
        if result["distance"] < threshold
    ]


def format_citation(result):
    metadata = result["metadata"]
    citation_parts = [
        metadata.get("source") or metadata.get("document"),
        metadata.get("section_number"),
        metadata.get("title"),
    ]
    citation_parts = [
        str(part).strip()
        for part in citation_parts
        if part is not None and str(part).strip()
    ]
    if citation_parts:
        return " | ".join(dict.fromkeys(citation_parts))
    return f'{result["database"]} | {result["id"]}'

# ---------------- Prompting ----------------------

SYSTEM_PROMPT = (
    "You are an EPLC/HHS domain assistant. Answer ONLY using information in the CONTEXT. "
    "If the CONTEXT cannot answer the question, reply exactly: Not specified in the provided context."
)

FALLBACK_PROMPT = (
    "You are a general expert assistant. Give a correct and helpful answer using general knowledge. "
    "DO NOT reference or imply context. DO NOT hallucinate context."
)

oa = OpenAI(api_key=OPENAI_API_KEY)

def make_prompt(q, docs):
    ctx = "\n\n---\n\n".join(docs)
    return f"CONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n"

def ask_openai(prompt, allow_fallback=False):
    sys_prompt = FALLBACK_PROMPT if allow_fallback else SYSTEM_PROMPT
    try:
        resp = oa.responses.create(
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        return f"[openai error] {e}"

def main():
    print(
        f"[ready] GPT={CHAT_MODEL} | embedding={EMBEDDING_MODEL} "
        f"| top_k={TOP_K}"
    )
    print("Ask EPLC/HHS questions. Type exit to quit.")

    while True:
        q = input("\nQ> ").strip()
        if q.lower() in ("exit","quit"):
            break
        print("Processing...")

        retrieved_results = retrieve(q, TOP_K)
        relevant_results = filter_relevant_results(retrieved_results)

        if not relevant_results:
            print("[debug] No valid domain context → strict refusal.")
            print("A> Not specified in the provided context.")
            print("   citations: []")
            continue

        context_documents = [
            result["document"]
            for result in relevant_results
        ]
        citations = [
            format_citation(result)
            for result in relevant_results
        ]

        prompt = make_prompt(q, context_documents)
        answer = ask_openai(prompt, allow_fallback=False)


        if answer.strip().lower() == "not specified in the provided context.":
            print("[debug] Context exists but insufficient → fallback general knowledge.")
            answer = ask_openai(q, allow_fallback=True)
            print("A>", answer)
            print("   citations: [] (general-knowledge fallback)")
            continue


        print("A>", answer)
        print("   citations:")
        for citation in citations:
            print(f"   - {citation}")


if __name__ == "__main__":
    main()
