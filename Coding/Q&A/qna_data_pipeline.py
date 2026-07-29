"""Prepare and build the versioned Q&A retrieval database.

The pipeline reads the cleaned source JSON files, creates traceable chunks,
and optionally embeds them into a new Chroma database. Existing databases are
never overwritten.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PREPARED_PATH = (
    REPOSITORY_ROOT / "Data" / "Q&A Processed" / "chunks_v4.json"
)
DEFAULT_DATABASE_PATH = (
    REPOSITORY_ROOT / "Data" / "Vector DataBase" / "qna_v4_bge_base_en_v1_5"
)
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_COLLECTION = "qna_knowledge_v4"
DEFAULT_MAX_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40
SCHEMA_VERSION = "4.0"

EPLC_SOURCE_FILES = (
    "Requirements_Analysis_Phase_3.4.json",
    "Design_Phase_3.5_MERGED_FINAL_FIXED_1765597020.json",
    "Development_Phase_3.6_MERGED_FINAL_1765598996.json",
    "Implementation_Phase_3.8.json",
)


def count_words(text):
    return len(text.split())


def repository_relative_path(path):
    """Return a portable repository-relative path when possible."""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved_path)


def normalize_text(text):
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text or "")
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs)


def sentence_units(text):
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units


def split_oversized_unit(unit, max_words):
    words = unit.split()
    return [
        " ".join(words[start:start + max_words])
        for start in range(0, len(words), max_words)
    ]


def trailing_overlap(units, overlap_words):
    overlap = []
    word_count = 0
    for unit in reversed(units):
        unit_words = count_words(unit)
        if word_count + unit_words > overlap_words:
            break
        overlap.insert(0, unit)
        word_count += unit_words
    return overlap


def chunk_text(text, max_words=DEFAULT_MAX_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS):
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words.")

    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        units = []
        for unit in sentence_units(paragraph):
            if count_words(unit) > max_words:
                units.extend(split_oversized_unit(unit, max_words))
            else:
                units.append(unit)

        current_units = []
        current_words = 0
        for unit in units:
            unit_words = count_words(unit)
            if current_units and current_words + unit_words > max_words:
                chunks.append(" ".join(current_units))
                current_units = trailing_overlap(
                    current_units,
                    overlap_words,
                )
                current_words = sum(
                    count_words(item)
                    for item in current_units
                )

            if current_units and current_words + unit_words > max_words:
                current_units = []
                current_words = 0

            current_units.append(unit)
            current_words += unit_words

        if current_units:
            paragraph_chunk = " ".join(current_units)
            if not chunks or paragraph_chunk != chunks[-1]:
                chunks.append(paragraph_chunk)

    return chunks


def stable_chunk_id(document_title, section_number, chunk_index, text):
    identity = (
        f"{document_title}|{section_number}|{chunk_index}|{text}"
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return f"qna-{section_number.replace('.', '-')}-{chunk_index:02d}-{digest}"


def overview_label(chunk_content):
    first_sentence = re.split(r"(?<=[.!?])\s+", chunk_content, maxsplit=1)[0]
    if ":" in first_sentence:
        label = first_sentence.split(":", 1)[0].strip()
        if 0 < count_words(label) <= 12:
            return label

    words = first_sentence.split()
    if len(words) > 18:
        return " ".join(words[:18]) + "..."
    return first_sentence


def section_overview(content_chunks):
    labels = []
    for chunk_content in content_chunks:
        label = overview_label(chunk_content)
        if label and label not in labels:
            labels.append(label)
    if len(labels) < 2:
        return ""
    return "This section covers: " + "; ".join(labels) + "."


def section_to_chunks(
    *,
    document_title,
    source_file,
    source_type,
    section_number,
    section_title,
    content,
    phase="",
    max_words=DEFAULT_MAX_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS,
):
    content_chunks = chunk_text(content, max_words, overlap_words)
    chunks = []
    overview = section_overview(content_chunks)
    total_chunk_count = len(content_chunks) + (1 if overview else 0)

    if overview:
        header = (
            f"{document_title}\n"
            f"Section {section_number}: {section_title}"
        )
        chunks.append(
            {
                "id": stable_chunk_id(
                    document_title,
                    str(section_number),
                    0,
                    overview,
                ),
                "document": f"{header}\n\n{overview}",
                "metadata": {
                    "document_title": document_title,
                    "source_file": source_file,
                    "source_type": source_type,
                    "section_number": str(section_number),
                    "section_title": section_title,
                    "chunk_index": 0,
                    "chunk_count": total_chunk_count,
                    "chunk_kind": "section_overview",
                    "content_word_count": count_words(overview),
                    "schema_version": SCHEMA_VERSION,
                    **({"phase": phase} if phase else {}),
                },
            }
        )

    for chunk_index, chunk_content in enumerate(content_chunks, start=1):
        header = (
            f"{document_title}\n"
            f"Section {section_number}: {section_title}"
        )
        document = f"{header}\n\n{chunk_content}"
        metadata = {
            "document_title": document_title,
            "source_file": source_file,
            "source_type": source_type,
            "section_number": str(section_number),
            "section_title": section_title,
            "chunk_index": chunk_index,
            "chunk_count": total_chunk_count,
            "chunk_kind": "section_content",
            "content_word_count": count_words(chunk_content),
            "schema_version": SCHEMA_VERSION,
        }
        if phase:
            metadata["phase"] = phase

        chunks.append(
            {
                "id": stable_chunk_id(
                    document_title,
                    str(section_number),
                    chunk_index,
                    chunk_content,
                ),
                "document": document,
                "metadata": metadata,
            }
        )
    return chunks


def load_eplc_chunks(
    source_directory,
    max_words=DEFAULT_MAX_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS,
):
    chunks = []
    source_files = []

    for filename in EPLC_SOURCE_FILES:
        path = source_directory / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(path)
        phase = data["section_title"]

        for subsection in data["subsections"]:
            chunks.extend(
                section_to_chunks(
                    document_title="HHS EPLC Framework",
                    source_file=filename,
                    source_type="eplc_framework",
                    phase=phase,
                    section_number=subsection["number"],
                    section_title=subsection["title"],
                    content=subsection["content"],
                    max_words=max_words,
                    overlap_words=overlap_words,
                )
            )

    return chunks, source_files


def load_hhs_policy_chunks(
    source_file,
    max_words=DEFAULT_MAX_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS,
):
    data = json.loads(source_file.read_text(encoding="utf-8"))
    chunks = []

    for section in data["sections"]:
        chunks.extend(
            section_to_chunks(
                document_title=data["document_title"],
                source_file=source_file.name,
                source_type="hhs_policy",
                section_number=section["number"],
                section_title=section["title"],
                content=section.get("content", ""),
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )
        for subsection in section.get("subsections", []):
            chunks.extend(
                section_to_chunks(
                    document_title=data["document_title"],
                    source_file=source_file.name,
                    source_type="hhs_policy",
                    section_number=subsection["number"],
                    section_title=subsection["title"],
                    content=subsection.get("content", ""),
                    max_words=max_words,
                    overlap_words=overlap_words,
                )
            )

    return chunks, [source_file]


def validate_chunks(chunks, max_words=DEFAULT_MAX_WORDS):
    if not chunks:
        raise ValueError("The prepared dataset contains no chunks.")

    required_metadata = {
        "document_title",
        "source_file",
        "source_type",
        "section_number",
        "section_title",
        "chunk_index",
        "chunk_count",
        "chunk_kind",
        "content_word_count",
        "schema_version",
    }
    seen_ids = set()

    for chunk in chunks:
        if not chunk["document"].strip():
            raise ValueError(f"Chunk {chunk['id']} has no document text.")
        if chunk["id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk['id']}")
        seen_ids.add(chunk["id"])

        missing = required_metadata - set(chunk["metadata"])
        if missing:
            raise ValueError(
                f"Chunk {chunk['id']} is missing metadata: {sorted(missing)}"
            )
        if chunk["metadata"]["content_word_count"] > max_words:
            raise ValueError(
                f"Chunk {chunk['id']} exceeds the {max_words}-word limit."
            )


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(
    repository_root=REPOSITORY_ROOT,
    max_words=DEFAULT_MAX_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS,
):
    repository_root = Path(repository_root)
    eplc_directory = (
        repository_root / "Data" / "EPLC Framework" / "EPLC Cleaned Data"
    )
    hhs_file = (
        repository_root
        / "Data"
        / "HHS EPLC Website"
        / "HHS EPLC Website.json"
    )

    eplc_chunks, eplc_files = load_eplc_chunks(
        eplc_directory,
        max_words,
        overlap_words,
    )
    hhs_chunks, hhs_files = load_hhs_policy_chunks(
        hhs_file,
        max_words,
        overlap_words,
    )
    chunks = eplc_chunks + hhs_chunks
    validate_chunks(chunks, max_words)

    source_files = eplc_files + hhs_files
    return {
        "schema_version": SCHEMA_VERSION,
        "chunking": {
            "strategy": "paragraph_preserving_sentence_window",
            "max_words": max_words,
            "overlap_words": overlap_words,
        },
        "sources": [
            {
                "path": str(path.relative_to(repository_root)),
                "sha256": file_sha256(path),
            }
            for path in source_files
        ],
        "statistics": {
            "chunk_count": len(chunks),
            "eplc_chunk_count": len(eplc_chunks),
            "hhs_policy_chunk_count": len(hhs_chunks),
        },
        "chunks": chunks,
    }


def write_prepared_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_chroma_database(
    prepared_path,
    output_path,
    model_name=DEFAULT_MODEL,
    collection_name=DEFAULT_COLLECTION,
    batch_size=16,
):
    prepared_path = Path(prepared_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing database: {output_path}"
        )

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    dataset = json.loads(prepared_path.read_text(encoding="utf-8"))
    chunks = dataset["chunks"]
    validate_chunks(
        chunks,
        dataset["chunking"]["max_words"],
    )

    model = SentenceTransformer(model_name, device="cpu")
    documents = [chunk["document"] for chunk in chunks]
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    output_path.mkdir(parents=True)
    client = PersistentClient(path=str(output_path))
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "schema_version": SCHEMA_VERSION,
        },
    )
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    embedding_records = [
        {
            "id": chunk["id"],
            "document": chunk["document"],
            "metadata": chunk["metadata"],
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    (output_path / "embeddings.json").write_text(
        json.dumps(embedding_records, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collection_name": collection_name,
        "embedding_model": model_name,
        "distance_metric": "cosine",
        "embedding_dimension": len(embeddings[0]),
        "chunk_count": len(chunks),
        "prepared_dataset": repository_relative_path(prepared_path),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare or build the versioned Q&A database."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREPARED_PATH,
    )
    prepare_parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
    )
    prepare_parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
    )

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--prepared",
        type=Path,
        default=DEFAULT_PREPARED_PATH,
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    build_parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        dataset = prepare_dataset(
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
        write_prepared_dataset(dataset, args.output)
        print(
            f"Prepared {dataset['statistics']['chunk_count']} chunks "
            f"at {args.output}"
        )
        return 0

    manifest = build_chroma_database(
        args.prepared,
        args.output,
        args.model,
        args.collection,
        args.batch_size,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
