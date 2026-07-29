"""Evaluate retrieval quality against the checked-in gold questions."""

import argparse
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_QUESTIONS = (
    REPOSITORY_ROOT / "evaluation" / "qna_retrieval_questions.json"
)
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages:"
)
DEFAULT_SIMILARITY_THRESHOLD = 0.5
LEGACY_EPLC_DIRECTORY = REPOSITORY_ROOT / "Data" / "EPLC New Embedding"
LEGACY_EPLC_FILES = (
    "Requirement_phase_embeddings_final.json",
    "design_phase_embeddings_final.json",
    "development_phase_embeddings_final.json",
    "Implementation_phase_embeddings_final.json",
)
LEGACY_HHS_FILE = (
    REPOSITORY_ROOT
    / "Data"
    / "HHS EPLC Website"
    / "HHS_EPLC_embeddings.json"
)


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(first, second):
    if len(first) != len(second):
        raise ValueError(
            f"Embedding dimension mismatch: {len(first)} != {len(second)}"
        )
    denominator = vector_norm(first) * vector_norm(second)
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(first, second)) / denominator


def load_questions(path=DEFAULT_QUESTIONS):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    seen_ids = set()
    for item in questions:
        required = {"id", "question", "expected_sections"}
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"Evaluation item is missing fields: {sorted(missing)}"
            )
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation ID: {item['id']}")
        seen_ids.add(item["id"])
        should_retrieve = item.get("should_retrieve", True)
        if should_retrieve and not item["expected_sections"]:
            raise ValueError(
                f"Evaluation item {item['id']} has no expected section."
            )
        if not should_retrieve and item["expected_sections"]:
            raise ValueError(
                f"Negative item {item['id']} cannot have expected sections."
            )
    return questions


def load_legacy_records(repository_root=REPOSITORY_ROOT):
    repository_root = Path(repository_root)
    eplc_directory = (
        repository_root / "Data" / "EPLC New Embedding"
    )
    hhs_file = (
        repository_root
        / "Data"
        / "HHS EPLC Website"
        / "HHS_EPLC_embeddings.json"
    )

    records = []
    for filename in LEGACY_EPLC_FILES:
        items = json.loads(
            (eplc_directory / filename).read_text(encoding="utf-8")
        )
        for index, item in enumerate(items):
            metadata = item["metadata"]
            records.append(
                {
                    "id": f"legacy-eplc-{filename}-{index}",
                    "document": metadata["content"],
                    "metadata": {
                        "source_type": "eplc_framework",
                        "source_file": filename,
                        "section_number": str(metadata["section_number"]),
                        "section_title": metadata["title"],
                        "legacy_type": metadata["type"],
                    },
                    "embedding": item["embedding"],
                }
            )

    hhs_items = json.loads(hhs_file.read_text(encoding="utf-8"))
    for index, item in enumerate(hhs_items):
        records.append(
            {
                "id": f"legacy-hhs-{index}",
                "document": item["text"],
                "metadata": {
                    "source_type": "hhs_policy",
                    "source_file": hhs_file.name,
                    "section_number": str(item["section_number"]),
                    "section_title": item["title"],
                },
                "embedding": item["embedding"],
            }
        )
    return records


def load_embedding_records(index_path):
    index_path = Path(index_path)
    if index_path.is_dir():
        index_path = index_path / "embeddings.json"
    records = json.loads(index_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Embedding index is empty.")
    return records


def rank_records(query_embedding, records):
    ranked = []
    for record in records:
        ranked.append(
            (
                cosine_similarity(
                    query_embedding,
                    record["embedding"],
                ),
                record,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def expected_rank(ranked_records, expected_sections):
    expected = {str(section) for section in expected_sections}
    for rank, (_, record) in enumerate(ranked_records, start=1):
        section_number = str(
            record["metadata"].get("section_number", "")
        )
        if section_number in expected:
            return rank
    return None


def collapse_by_section(ranked_records):
    collapsed = []
    seen_sections = set()
    for similarity, record in ranked_records:
        section_number = str(
            record["metadata"].get("section_number", "")
        )
        if section_number in seen_sections:
            continue
        seen_sections.add(section_number)
        collapsed.append((similarity, record))
    return collapsed


def evaluate_rankings(
    questions,
    query_embeddings,
    records,
    cutoffs=(1, 3, 5, 6),
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
):
    if len(questions) != len(query_embeddings):
        raise ValueError("Each evaluation question needs one embedding.")

    hits = {cutoff: 0 for cutoff in cutoffs}
    section_hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_rank_total = 0.0
    section_reciprocal_rank_total = 0.0
    positive_above_threshold = 0
    negative_below_threshold = 0
    positive_count = 0
    negative_count = 0
    details = []
    maximum_cutoff = max(cutoffs)

    for question, query_embedding in zip(questions, query_embeddings):
        ranked = rank_records(query_embedding, records)
        top_similarity = ranked[0][0]
        should_retrieve = question.get("should_retrieve", True)

        if not should_retrieve:
            negative_count += 1
            if top_similarity < similarity_threshold:
                negative_below_threshold += 1
            details.append(
                {
                    "id": question["id"],
                    "expected_sections": [],
                    "should_retrieve": False,
                    "top_similarity": round(top_similarity, 4),
                    "would_retrieve": (
                        top_similarity >= similarity_threshold
                    ),
                    "top_sections": [
                        str(
                            record["metadata"].get(
                                "section_number",
                                "",
                            )
                        )
                        for _, record in ranked[:maximum_cutoff]
                    ],
                }
            )
            continue

        positive_count += 1
        rank = expected_rank(
            ranked,
            question["expected_sections"],
        )
        section_ranked = collapse_by_section(ranked)
        section_rank = expected_rank(
            section_ranked,
            question["expected_sections"],
        )
        if rank is not None:
            reciprocal_rank_total += 1 / rank
            relevant_similarity = ranked[rank - 1][0]
            if relevant_similarity >= similarity_threshold:
                positive_above_threshold += 1
            for cutoff in cutoffs:
                if rank <= cutoff:
                    hits[cutoff] += 1
        else:
            relevant_similarity = None
        if section_rank is not None:
            section_reciprocal_rank_total += 1 / section_rank
            for cutoff in cutoffs:
                if section_rank <= cutoff:
                    section_hits[cutoff] += 1

        details.append(
            {
                "id": question["id"],
                "expected_sections": question["expected_sections"],
                "should_retrieve": True,
                "first_relevant_rank": rank,
                "first_relevant_section_rank": section_rank,
                "top_similarity": round(top_similarity, 4),
                "relevant_similarity": (
                    round(relevant_similarity, 4)
                    if relevant_similarity is not None
                    else None
                ),
                "top_sections": [
                    str(
                        record["metadata"].get(
                            "section_number",
                            "",
                        )
                    )
                    for _, record in ranked[:maximum_cutoff]
                ],
            }
        )

    question_count = len(questions)
    positive_recall = (
        positive_above_threshold / positive_count
        if positive_count
        else 0.0
    )
    negative_specificity = (
        negative_below_threshold / negative_count
        if negative_count
        else 0.0
    )
    threshold_candidates = []
    for candidate_index in range(40, 81):
        candidate = candidate_index / 100
        candidate_positive_recall = (
            sum(
                detail.get("relevant_similarity", -1) >= candidate
                for detail in details
                if detail["should_retrieve"]
            ) / positive_count
            if positive_count
            else 0.0
        )
        candidate_negative_specificity = (
            sum(
                detail["top_similarity"] < candidate
                for detail in details
                if not detail["should_retrieve"]
            ) / negative_count
            if negative_count
            else 0.0
        )
        threshold_candidates.append(
            {
                "similarity_threshold": candidate,
                "positive_recall": candidate_positive_recall,
                "negative_specificity": candidate_negative_specificity,
                "balanced_accuracy": (
                    candidate_positive_recall
                    + candidate_negative_specificity
                ) / 2,
            }
        )
    best_threshold = max(
        threshold_candidates,
        key=lambda candidate: (
            candidate["balanced_accuracy"],
            candidate["negative_specificity"],
            candidate["similarity_threshold"],
        ),
    )

    return {
        "question_count": question_count,
        "positive_question_count": positive_count,
        "negative_question_count": negative_count,
        "metrics": {
            **{
                f"recall_at_{cutoff}": round(
                    hits[cutoff] / positive_count,
                    4,
                )
                for cutoff in cutoffs
            },
            "mrr": round(
                reciprocal_rank_total / positive_count,
                4,
            ),
        },
        "section_metrics": {
            **{
                f"recall_at_{cutoff}": round(
                    section_hits[cutoff] / positive_count,
                    4,
                )
                for cutoff in cutoffs
            },
            "mrr": round(
                section_reciprocal_rank_total / positive_count,
                4,
            ),
        },
        "threshold_metrics": {
            "similarity_threshold": similarity_threshold,
            "positive_recall": round(positive_recall, 4),
            "negative_specificity": round(negative_specificity, 4),
            "false_positive_rate": round(1 - negative_specificity, 4),
            "balanced_accuracy": round(
                (positive_recall + negative_specificity) / 2,
                4,
            ),
        },
        "recommended_threshold": {
            key: round(value, 4)
            for key, value in best_threshold.items()
        },
        "details": details,
    }


def embed_questions(
    questions,
    model_name=DEFAULT_MODEL,
    query_instruction=DEFAULT_QUERY_INSTRUCTION,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    instructed_questions = [
        f"{query_instruction.rstrip()} {item['question']}"
        for item in questions
    ]
    return model.encode(
        instructed_questions,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an embedding index with gold questions."
    )
    parser.add_argument(
        "--index",
        default="legacy",
        help="'legacy', an embeddings.json file, or a built DB directory.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--query-instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    questions = load_questions(args.questions)
    records = (
        load_legacy_records()
        if args.index == "legacy"
        else load_embedding_records(args.index)
    )
    query_embeddings = embed_questions(
        questions,
        args.model,
        args.query_instruction,
    )
    report = evaluate_rankings(
        questions,
        query_embeddings,
        records,
        similarity_threshold=args.similarity_threshold,
    )
    report["index"] = args.index
    report["model"] = args.model

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
