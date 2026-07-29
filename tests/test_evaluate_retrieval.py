import importlib.util
from pathlib import Path
from unittest import TestCase


EVALUATOR_FILE = (
    Path(__file__).parents[1]
    / "Coding"
    / "Q&A"
    / "evaluate_retrieval.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "evaluate_retrieval",
    EVALUATOR_FILE,
)
evaluator = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(evaluator)


def record(section_number, embedding):
    return {
        "id": f"section-{section_number}",
        "document": "text",
        "metadata": {"section_number": section_number},
        "embedding": embedding,
    }


class EvaluateRetrievalTests(TestCase):
    def test_gold_questions_are_valid(self):
        questions = evaluator.load_questions()

        self.assertEqual(34, len(questions))
        self.assertEqual(
            len(questions),
            len({question["id"] for question in questions}),
        )

    def test_legacy_index_is_complete(self):
        records = evaluator.load_legacy_records()

        self.assertEqual(55, len(records))
        self.assertTrue(
            all(len(item["embedding"]) == 1024 for item in records)
        )

    def test_cosine_similarity_is_scale_independent(self):
        similarity = evaluator.cosine_similarity(
            [1.0, 0.0],
            [5.0, 0.0],
        )

        self.assertAlmostEqual(1.0, similarity)

    def test_metrics_use_expected_section_rank(self):
        questions = [
            {
                "id": "first",
                "question": "first",
                "expected_sections": ["1"],
            },
            {
                "id": "second",
                "question": "second",
                "expected_sections": ["2"],
            },
        ]
        records = [
            record("1", [1.0, 0.0]),
            record("2", [0.0, 1.0]),
            record("3", [0.7, 0.7]),
        ]
        query_embeddings = [
            [1.0, 0.0],
            [1.0, 0.1],
        ]

        report = evaluator.evaluate_rankings(
            questions,
            query_embeddings,
            records,
        )

        self.assertEqual(0.5, report["metrics"]["recall_at_1"])
        self.assertEqual(1.0, report["metrics"]["recall_at_3"])
        self.assertEqual(0.6667, report["metrics"]["mrr"])

    def test_section_metrics_do_not_count_duplicate_chunks_twice(self):
        questions = [
            {
                "id": "target",
                "question": "target",
                "expected_sections": ["2"],
            }
        ]
        records = [
            record("1", [1.0, 0.0]),
            record("1", [0.9, 0.1]),
            record("2", [0.8, 0.2]),
        ]

        report = evaluator.evaluate_rankings(
            questions,
            [[1.0, 0.0]],
            records,
            cutoffs=(1, 2, 3),
        )

        self.assertEqual(0.0, report["metrics"]["recall_at_2"])
        self.assertEqual(1.0, report["section_metrics"]["recall_at_2"])

    def test_negative_questions_measure_threshold_specificity(self):
        questions = [
            {
                "id": "positive",
                "question": "positive",
                "expected_sections": ["1"],
            },
            {
                "id": "negative",
                "question": "negative",
                "expected_sections": [],
                "should_retrieve": False,
            },
        ]
        records = [
            record("1", [1.0, 0.0]),
            record("2", [0.0, 1.0]),
        ]

        report = evaluator.evaluate_rankings(
            questions,
            [[1.0, 0.0], [0.7, 0.7]],
            records,
            similarity_threshold=0.8,
        )

        self.assertEqual(
            1.0,
            report["threshold_metrics"]["positive_recall"],
        )
        self.assertEqual(
            1.0,
            report["threshold_metrics"]["negative_specificity"],
        )

    def test_dimension_mismatch_is_explicit(self):
        with self.assertRaisesRegex(
            ValueError,
            "dimension mismatch",
        ):
            evaluator.cosine_similarity([1.0], [1.0, 2.0])
