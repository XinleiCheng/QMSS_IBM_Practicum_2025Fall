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

coll_eplc = get_single_collection(eplc_db, "EPLCFramework_db")
coll_hhs  = get_single_collection(hhs_db, "HHS_db")


# ---------------- Retrieval ----------------------

def retrieve_exact(substring: str, k: int = TOP_K):
    ids, docs = [], []
    for coll in (coll_eplc, coll_hhs):
        try:
            r = coll.get(
                where_document={"$contains": substring},
                include=["documents"],
                limit=k,
            )
        except:
            continue
        ids.extend(r.get("ids", []))
        docs.extend(r.get("documents", []))
    return ids[:k], docs[:k], [0.0]*min(len(docs),k)

def retrieve(query: str, k: int = TOP_K):
    instructed_query = f"{QUERY_INSTRUCTION}{query}"
    qv = sbert.encode(
        [instructed_query],
        normalize_embeddings=True,
    ).tolist()

    # EPLC
    r1 = coll_eplc.query(query_embeddings=qv, n_results=k,
                         include=["documents", "distances"])
    ids1  = r1.get("ids", [[]])[0]
    docs1 = r1.get("documents", [[]])[0]
    dist1 = r1.get("distances", [[]])[0]

    # HHS
    r2 = coll_hhs.query(query_embeddings=qv, n_results=k,
                         include=["documents", "distances"])
    ids2  = r2.get("ids", [[]])[0]
    docs2 = r2.get("documents", [[]])[0]
    dist2 = r2.get("distances", [[]])[0]

    combined = []
    for a,b,c in zip(ids1,docs1,dist1): combined.append((a,b,c))
    for a,b,c in zip(ids2,docs2,dist2): combined.append((a,b,c))

    combined.sort(key=lambda x: x[2])
    combined = combined[:k]

    return [x[0] for x in combined], [x[1] for x in combined], [x[2] for x in combined]


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

        # 1. exact retrieval
        ids_exact, docs_exact, _ = retrieve_exact(q, TOP_K)

        # 2. semantic retrieval
        ids_sem, docs_sem, dists_sem = retrieve(q, TOP_K)

        # ---- Semantic relevance filter ----
        sem_valid = [
            (i, d) for i, d, dist in zip(ids_sem, docs_sem, dists_sem)
            if dist < SEM_THRESHOLD
        ]
        ids_sem  = [x[0] for x in sem_valid]
        docs_sem = [x[1] for x in sem_valid]

        # combine
        combined_ids  = ids_exact + ids_sem
        combined_docs = docs_exact + docs_sem


        if not combined_docs:
            print("[debug] No valid domain context → strict refusal.")
            print("A> Not specified in the provided context.")
            print("   citations: []")
            continue


        prompt = make_prompt(q, combined_docs)
        answer = ask_openai(prompt, allow_fallback=False)


        if answer.strip().lower() == "not specified in the provided context.":
            print("[debug] Context exists but insufficient → fallback general knowledge.")
            answer = ask_openai(q, allow_fallback=True)
            print("A>", answer)
            print("   citations:", combined_ids)
            continue


        print("A>", answer)
        print("   citations:", combined_ids)


if __name__ == "__main__":
    main()
