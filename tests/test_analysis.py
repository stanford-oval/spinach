"""
Tests for spinach_agent.analysis module.

Covers:
  - SPARQL complexity analysis
  - Per-example precision / recall / F1 breakdown
  - Error categorization
  - AnalysisReport aggregation
  - Cross-run BenchmarkComparison
  - JSON / CSV export round-trips
"""

import json
import math
import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.insert(0, parent_directory)

from spinach_agent.analysis import (
    AnalysisReport,
    BenchmarkComparison,
    ExampleMetrics,
    SparqlComplexity,
    analyse_sparql_complexity,
    categorize_error,
    compare_runs,
    compute_example_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(val, places=6):
    factor = 10 ** places
    return math.ceil(val * factor) / factor


def _make_entry(
    question="test?",
    predicted_sparql="SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }",
    gold_sparql="SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }",
    predicted_answer=None,
    gold_answer=None,
    f1=1.0,
):
    return {
        "question": question,
        "predicted_sparql": predicted_sparql,
        "gold_sparql": gold_sparql,
        "predicted_answer_tuple": predicted_answer or [],
        "gold_answer_tuple": gold_answer or [],
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# 1. SPARQL Complexity Tests
# ---------------------------------------------------------------------------

class TestSparqlComplexity(unittest.TestCase):
    def test_empty_sparql(self):
        c = analyse_sparql_complexity(None)
        self.assertEqual(c.complexity_score, 0.0)
        self.assertEqual(c.complexity_bucket, "simple")

    def test_simple_select(self):
        sparql = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        c = analyse_sparql_complexity(sparql)
        self.assertTrue(c.is_select)
        self.assertFalse(c.is_ask)
        self.assertGreater(c.num_triples, 0)
        self.assertEqual(c.complexity_bucket, "simple")

    def test_ask_query(self):
        sparql = "ASK WHERE { wd:Q76 wdt:P31 wd:Q5 }"
        c = analyse_sparql_complexity(sparql)
        self.assertTrue(c.is_ask)

    def test_complex_query(self):
        sparql = """
        SELECT DISTINCT ?x ?xLabel WHERE {
            ?x wdt:P31 wd:Q5 .
            ?x wdt:P27 wd:Q30 .
            ?x wdt:P106 wd:Q82955 .
            OPTIONAL { ?x wdt:P569 ?dob }
            FILTER(?dob > "1990-01-01"^^xsd:dateTime)
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        GROUP BY ?x ?xLabel
        ORDER BY ?xLabel
        LIMIT 100
        """
        c = analyse_sparql_complexity(sparql)
        self.assertTrue(c.has_distinct)
        self.assertGreater(c.num_optionals, 0)
        self.assertGreater(c.num_filters, 0)
        self.assertTrue(c.has_group_by)
        self.assertTrue(c.has_order_by)
        self.assertTrue(c.has_limit)
        self.assertGreater(c.num_services, 0)
        self.assertIn(c.complexity_bucket, ("medium", "complex", "very_complex"))

    def test_union_query(self):
        sparql = """
        SELECT ?x WHERE {
            { ?x wdt:P31 wd:Q5 } UNION { ?x wdt:P31 wd:Q515 }
        }
        """
        c = analyse_sparql_complexity(sparql)
        self.assertGreater(c.num_unions, 0)

    def test_subquery(self):
        sparql = """
        SELECT ?x WHERE {
            { SELECT ?x WHERE { ?x wdt:P31 wd:Q5 } LIMIT 10 }
            ?x wdt:P27 wd:Q30 .
        }
        """
        c = analyse_sparql_complexity(sparql)
        self.assertGreater(c.num_subqueries, 0)

    def test_property_path(self):
        sparql = "SELECT ?x WHERE { ?x wdt:P31/wdt:P279 wd:Q5 }"
        c = analyse_sparql_complexity(sparql)
        self.assertGreater(c.num_property_paths, 0)

    def test_minus(self):
        sparql = """
        SELECT ?x WHERE {
            ?x wdt:P31 wd:Q5 .
            MINUS { ?x wdt:P570 ?death }
        }
        """
        c = analyse_sparql_complexity(sparql)
        self.assertGreater(c.num_minus, 0)

    def test_aggregate(self):
        sparql = "SELECT (COUNT(?x) AS ?count) WHERE { ?x wdt:P31 wd:Q5 }"
        c = analyse_sparql_complexity(sparql)
        self.assertGreater(c.num_aggregates, 0)

    def test_complexity_score_monotonic(self):
        """More features → higher score."""
        s1 = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        s2 = """
        SELECT DISTINCT ?x WHERE {
            ?x wdt:P31 wd:Q5 .
            ?x wdt:P27 wd:Q30 .
            OPTIONAL { ?x wdt:P569 ?dob }
            FILTER(?dob > "1990-01-01"^^xsd:dateTime)
        }
        GROUP BY ?x
        """
        c1 = analyse_sparql_complexity(s1)
        c2 = analyse_sparql_complexity(s2)
        self.assertGreater(c2.complexity_score, c1.complexity_score)


# ---------------------------------------------------------------------------
# 2. Error Categorization Tests
# ---------------------------------------------------------------------------

class TestErrorCategorization(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(
            categorize_error("SELECT ...", [{"a": 1}], [{"a": 1}], 1.0),
            "correct",
        )

    def test_no_sparql(self):
        self.assertEqual(
            categorize_error(None, [], [{"a": 1}], 0.0),
            "no_sparql",
        )

    def test_empty_prediction(self):
        self.assertEqual(
            categorize_error("SELECT ...", [], [{"a": 1}], 0.0),
            "empty_prediction",
        )

    def test_high_partial(self):
        self.assertEqual(
            categorize_error("SELECT ...", [{"a": 1}], [{"a": 1}], 0.7),
            "high_partial",
        )

    def test_low_partial(self):
        self.assertEqual(
            categorize_error("SELECT ...", [{"a": 1}], [{"a": 1}], 0.3),
            "low_partial",
        )

    def test_zero_f1(self):
        self.assertEqual(
            categorize_error("SELECT ...", [{"a": 1}], [{"b": 2}], 0.0),
            "zero_f1",
        )


# ---------------------------------------------------------------------------
# 3. Per-Example Metrics Tests
# ---------------------------------------------------------------------------

class TestExampleMetrics(unittest.TestCase):
    def test_perfect_match(self):
        entry = _make_entry(
            predicted_answer=[
                {"item": {"type": "uri", "value": "a"}}
            ],
            gold_answer=[
                {"item": {"type": "uri", "value": "a"}}
            ],
            f1=1.0,
        )
        m = compute_example_metrics(entry)
        self.assertTrue(m.is_exact_match)
        self.assertEqual(m.error_category, "correct")
        self.assertAlmostEqual(m.precision, 1.0, places=4)
        self.assertAlmostEqual(m.recall, 1.0, places=4)

    def test_no_prediction(self):
        entry = _make_entry(
            predicted_sparql=None,
            predicted_answer=[],
            gold_answer=[
                {"item": {"type": "uri", "value": "a"}}
            ],
            f1=0.0,
        )
        m = compute_example_metrics(entry)
        self.assertFalse(m.is_exact_match)
        self.assertEqual(m.error_category, "no_sparql")

    def test_complexity_attached(self):
        entry = _make_entry(
            predicted_sparql="SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }",
            gold_sparql="SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q5 . OPTIONAL { ?x wdt:P569 ?d } }",
        )
        m = compute_example_metrics(entry)
        self.assertIsNotNone(m.predicted_complexity)
        self.assertIsNotNone(m.gold_complexity)
        self.assertGreaterEqual(
            m.gold_complexity.complexity_score,
            m.predicted_complexity.complexity_score,
        )


# ---------------------------------------------------------------------------
# 4. AnalysisReport Tests
# ---------------------------------------------------------------------------

class TestAnalysisReport(unittest.TestCase):
    def _make_sample_entries(self):
        return [
            _make_entry(
                question="Q1",
                predicted_answer=[{"item": {"type": "uri", "value": "a"}}],
                gold_answer=[{"item": {"type": "uri", "value": "a"}}],
                f1=1.0,
            ),
            _make_entry(
                question="Q2",
                predicted_answer=[{"item": {"type": "uri", "value": "a"}}],
                gold_answer=[
                    {"item": {"type": "uri", "value": "a"}},
                    {"item": {"type": "uri", "value": "b"}},
                ],
                f1=0.5,
            ),
            _make_entry(
                question="Q3",
                predicted_sparql=None,
                predicted_answer=[],
                gold_answer=[{"item": {"type": "uri", "value": "c"}}],
                f1=0.0,
            ),
        ]

    def test_from_entries(self):
        entries = self._make_sample_entries()
        report = AnalysisReport.from_entries(entries)
        self.assertEqual(report.total, 3)
        self.assertAlmostEqual(report.macro_f1, (1.0 + 0.5 + 0.0) / 3, places=4)
        self.assertIn("correct", report.error_distribution)

    def test_summary_runs(self):
        entries = self._make_sample_entries()
        report = AnalysisReport.from_entries(entries)
        summary = report.summary()
        self.assertIn("Macro F1", summary)
        self.assertIn("Error Distribution", summary)

    def test_json_round_trip(self):
        entries = self._make_sample_entries()
        report = AnalysisReport.from_entries(entries, source_file="test.json")

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            report.save_json(tmp_path)
            with open(tmp_path, "r") as fd:
                loaded = json.load(fd)
            self.assertEqual(loaded["total"], 3)
            self.assertEqual(len(loaded["examples"]), 3)
            self.assertIn("macro_f1", loaded)
            self.assertIn("error_distribution", loaded)
        finally:
            os.unlink(tmp_path)

    def test_csv_export(self):
        entries = self._make_sample_entries()
        report = AnalysisReport.from_entries(entries)

        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            report.save_csv(tmp_path)
            with open(tmp_path, "r") as fd:
                lines = fd.readlines()
            # header + 3 data rows
            self.assertEqual(len(lines), 4)
        finally:
            os.unlink(tmp_path)

    def test_from_file(self):
        entries = self._make_sample_entries()
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(entries, tmp, ensure_ascii=False)
            tmp_path = tmp.name

        try:
            report = AnalysisReport.from_file(tmp_path)
            self.assertEqual(report.total, 3)
        finally:
            os.unlink(tmp_path)

    def test_empty_entries(self):
        report = AnalysisReport.from_entries([])
        self.assertEqual(report.total, 0)
        self.assertEqual(report.macro_f1, 0.0)


# ---------------------------------------------------------------------------
# 5. BenchmarkComparison Tests
# ---------------------------------------------------------------------------

class TestBenchmarkComparison(unittest.TestCase):
    def test_comparison_table(self):
        entries1 = [
            _make_entry(
                predicted_answer=[{"item": {"type": "uri", "value": "a"}}],
                gold_answer=[{"item": {"type": "uri", "value": "a"}}],
                f1=1.0,
            )
        ]
        entries2 = [
            _make_entry(
                predicted_answer=[],
                gold_answer=[{"item": {"type": "uri", "value": "a"}}],
                f1=0.0,
            )
        ]

        r1 = AnalysisReport.from_entries(entries1, source_file="run_a.json")
        r2 = AnalysisReport.from_entries(entries2, source_file="run_b.json")

        bench = BenchmarkComparison()
        bench.add(r1)
        bench.add(r2)

        table = bench.comparison_table()
        self.assertIn("run_a.json", table)
        self.assertIn("run_b.json", table)
        self.assertIn("F1", table)

    def test_json_export(self):
        entries = [
            _make_entry(
                predicted_answer=[{"item": {"type": "uri", "value": "a"}}],
                gold_answer=[{"item": {"type": "uri", "value": "a"}}],
                f1=1.0,
            )
        ]
        r = AnalysisReport.from_entries(entries, source_file="single.json")
        bench = BenchmarkComparison()
        bench.add(r)

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            bench.save_json(tmp_path)
            with open(tmp_path, "r") as fd:
                data = json.load(fd)
            self.assertIn("runs", data)
            self.assertEqual(len(data["runs"]), 1)
        finally:
            os.unlink(tmp_path)

    def test_compare_runs_helper(self):
        entries = [
            _make_entry(
                predicted_answer=[{"item": {"type": "uri", "value": "a"}}],
                gold_answer=[{"item": {"type": "uri", "value": "a"}}],
                f1=1.0,
            )
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as tmp1:
            json.dump(entries, tmp1, ensure_ascii=False)
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as tmp2:
            json.dump(entries, tmp2, ensure_ascii=False)
            path2 = tmp2.name

        try:
            bench = compare_runs(path1, path2)
            self.assertEqual(len(bench.reports), 2)
            table = bench.comparison_table()
            self.assertIn("F1", table)
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ---------------------------------------------------------------------------
# 6. Precision / Recall edge cases
# ---------------------------------------------------------------------------

class TestPrecisionRecall(unittest.TestCase):
    def test_boolean_match(self):
        from spinach_agent.analysis import _precision_recall_from_cost_matrix

        p, r = _precision_recall_from_cost_matrix(True, True)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)

    def test_boolean_mismatch(self):
        from spinach_agent.analysis import _precision_recall_from_cost_matrix

        p, r = _precision_recall_from_cost_matrix(True, False)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_none_prediction(self):
        from spinach_agent.analysis import _precision_recall_from_cost_matrix

        p, r = _precision_recall_from_cost_matrix(None, [{"x": {"type": "uri", "value": "a"}}])
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_empty_gold(self):
        from spinach_agent.analysis import _precision_recall_from_cost_matrix

        p, r = _precision_recall_from_cost_matrix(
            [{"x": {"type": "uri", "value": "a"}}], []
        )
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)


if __name__ == "__main__":
    unittest.main()

