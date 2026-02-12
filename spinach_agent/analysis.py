"""
Evaluation Analysis & Benchmarking Module for SPINACH.

Provides detailed evaluation metrics beyond F1 and EM:
  - Per-example precision, recall, and F1 breakdown
  - SPARQL complexity analysis (triple count, operators, depth)
  - Error categorization for failed predictions
  - Cross-run comparison and benchmarking utilities
  - Exportable analysis reports (JSON / CSV)

Usage (CLI):
    python spinach_agent/analysis.py \
        --input datasets/qald_10/en/spinach_output_test.json \
        --output analysis_report.json \
        --format json

Usage (Library):
    from spinach_agent.analysis import AnalysisReport
    report = AnalysisReport.from_file("spinach_output_test.json")
    print(report.summary())
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Path setup – mirrors the pattern used elsewhere in the repo
# ---------------------------------------------------------------------------
current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

# ---------------------------------------------------------------------------
# Inline the core metric helpers from eval.py to avoid pulling in chainlite
# (and its heavy transitive deps) at import time.  These functions are exact
# copies of the logic in eval.py so that numbers stay consistent.
# ---------------------------------------------------------------------------

def _compute_match_ratio(predicted, gold):
    """
    Computes per-row match ratio between a predicted row and a gold row.
    (Mirrored from eval.py)
    """
    from collections import Counter as _Counter

    gold_values = [
        tuple(sorted(gold_value.items()))
        for gold_key, gold_value in gold.items()
        if not (gold_key.endswith("Label") or gold_key.endswith("Description"))
    ]
    useful_predicted_values = [
        tuple(sorted(predicted_value.items()))
        for predicted_value in predicted.values()
        if tuple(sorted(predicted_value.items())) in gold_values
    ]

    overlap = sum((_Counter(gold_values) & _Counter(useful_predicted_values)).values())
    if len(gold_values) == 0:
        return 0
    return overlap / len(gold_values)


# ===================================================================
# 1. SPARQL Complexity Analysis
# ===================================================================

# Regex patterns for SPARQL analysis
_TRIPLE_PATTERN = re.compile(
    r"(?:wd:|wdt:|p:|ps:|pq:|pqv:|psv:|psn:|\?)\S+\s+"
    r"(?:wd:|wdt:|p:|ps:|pq:|pqv:|psv:|psn:|a\b|rdfs:\w+|schema:\w+|\?)\S+\s+"
    r"(?:wd:|wdt:|p:|ps:|pq:|pqv:|psv:|psn:|\?|\")",
    re.IGNORECASE,
)
_FILTER_PATTERN = re.compile(r"\bFILTER\s*\(", re.IGNORECASE)
_OPTIONAL_PATTERN = re.compile(r"\bOPTIONAL\s*\{", re.IGNORECASE)
_UNION_PATTERN = re.compile(r"\bUNION\s*\{", re.IGNORECASE)
_SUBQUERY_PATTERN = re.compile(
    r"\{\s*SELECT\b", re.IGNORECASE
)
_SERVICE_PATTERN = re.compile(r"\bSERVICE\b", re.IGNORECASE)
_GROUP_BY_PATTERN = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_HAVING_PATTERN = re.compile(r"\bHAVING\s*\(", re.IGNORECASE)
_VALUES_PATTERN = re.compile(r"\bVALUES\b", re.IGNORECASE)
_BIND_PATTERN = re.compile(r"\bBIND\s*\(", re.IGNORECASE)
_AGGREGATE_PATTERN = re.compile(
    r"\b(?:COUNT|SUM|AVG|MIN|MAX|SAMPLE|GROUP_CONCAT)\s*\(", re.IGNORECASE
)
_MINUS_PATTERN = re.compile(r"\bMINUS\s*\{", re.IGNORECASE)
_PROPERTY_PATH_PATTERN = re.compile(
    r"wdt:P\d+\s*/\s*wdt:P\d+|wdt:P\d+\s*\*|\^wdt:P\d+", re.IGNORECASE
)
_ASK_PATTERN = re.compile(r"\bASK\s+WHERE\b", re.IGNORECASE)
_SELECT_PATTERN = re.compile(r"\bSELECT\b", re.IGNORECASE)
_DISTINCT_PATTERN = re.compile(r"\bSELECT\s+DISTINCT\b", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_VARIABLE_PATTERN = re.compile(r"\?\w+")


@dataclass
class SparqlComplexity:
    """Measures of SPARQL query complexity."""

    num_triples: int = 0
    num_filters: int = 0
    num_optionals: int = 0
    num_unions: int = 0
    num_subqueries: int = 0
    num_services: int = 0
    has_group_by: bool = False
    has_order_by: bool = False
    has_having: bool = False
    has_values: bool = False
    num_binds: int = 0
    num_aggregates: int = 0
    num_minus: int = 0
    num_property_paths: int = 0
    is_ask: bool = False
    is_select: bool = False
    has_distinct: bool = False
    has_limit: bool = False
    num_variables: int = 0
    query_length: int = 0

    @property
    def complexity_score(self) -> float:
        """
        A single scalar summarising query complexity.

        Weights are intentionally transparent so researchers can adjust them.
        """
        return (
            self.num_triples * 1.0
            + self.num_filters * 1.5
            + self.num_optionals * 1.2
            + self.num_unions * 2.0
            + self.num_subqueries * 3.0
            + self.num_services * 0.5
            + (1.5 if self.has_group_by else 0)
            + (0.5 if self.has_order_by else 0)
            + (1.5 if self.has_having else 0)
            + (1.0 if self.has_values else 0)
            + self.num_binds * 1.0
            + self.num_aggregates * 1.5
            + self.num_minus * 1.5
            + self.num_property_paths * 2.0
        )

    @property
    def complexity_bucket(self) -> str:
        """Categorise as simple / medium / complex / very_complex."""
        s = self.complexity_score
        if s <= 3:
            return "simple"
        elif s <= 8:
            return "medium"
        elif s <= 15:
            return "complex"
        return "very_complex"


def analyse_sparql_complexity(sparql: Optional[str]) -> SparqlComplexity:
    """Compute complexity metrics for a single SPARQL string."""
    if not sparql:
        return SparqlComplexity()

    return SparqlComplexity(
        num_triples=len(_TRIPLE_PATTERN.findall(sparql)),
        num_filters=len(_FILTER_PATTERN.findall(sparql)),
        num_optionals=len(_OPTIONAL_PATTERN.findall(sparql)),
        num_unions=len(_UNION_PATTERN.findall(sparql)),
        num_subqueries=len(_SUBQUERY_PATTERN.findall(sparql)),
        num_services=len(_SERVICE_PATTERN.findall(sparql)),
        has_group_by=bool(_GROUP_BY_PATTERN.search(sparql)),
        has_order_by=bool(_ORDER_BY_PATTERN.search(sparql)),
        has_having=bool(_HAVING_PATTERN.search(sparql)),
        has_values=bool(_VALUES_PATTERN.search(sparql)),
        num_binds=len(_BIND_PATTERN.findall(sparql)),
        num_aggregates=len(_AGGREGATE_PATTERN.findall(sparql)),
        num_minus=len(_MINUS_PATTERN.findall(sparql)),
        num_property_paths=len(_PROPERTY_PATH_PATTERN.findall(sparql)),
        is_ask=bool(_ASK_PATTERN.search(sparql)),
        is_select=bool(_SELECT_PATTERN.search(sparql)),
        has_distinct=bool(_DISTINCT_PATTERN.search(sparql)),
        has_limit=bool(_LIMIT_PATTERN.search(sparql)),
        num_variables=len(set(_VARIABLE_PATTERN.findall(sparql))),
        query_length=len(sparql),
    )


# ===================================================================
# 2. Per-Example Precision / Recall / F1
# ===================================================================

@dataclass
class ExampleMetrics:
    """Detailed metrics for a single predicted-vs-gold pair."""

    question: str
    predicted_sparql: Optional[str]
    gold_sparql: Optional[str] = None
    predicted_answer: Optional[list] = None
    gold_answer: Optional[list] = None
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    is_exact_match: bool = False
    error_category: str = "none"
    predicted_complexity: Optional[SparqlComplexity] = None
    gold_complexity: Optional[SparqlComplexity] = None


def _precision_recall_from_cost_matrix(
    predicted_results: list, gold_results: list
) -> Tuple[float, float]:
    """
    Compute precision and recall using the same Hungarian-matching approach
    as the repo's ``f1()`` so numbers are consistent.
    """
    if predicted_results is None or not predicted_results:
        return (0.0, 0.0)
    if not gold_results:
        return (0.0, 0.0)
    if isinstance(predicted_results, bool) or isinstance(gold_results, bool):
        match = int(predicted_results == gold_results)
        return (float(match), float(match))

    cost_matrix = np.empty((len(predicted_results), len(gold_results)))
    for i in range(len(predicted_results)):
        for j in range(len(gold_results)):
            cost_matrix[i, j] = _compute_match_ratio(
                predicted_results[i], gold_results[j]
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    assigned = cost_matrix[row_ind, col_ind]

    tp = assigned[assigned > 0].sum()

    # Precision denominator = total predicted rows
    precision = tp / len(predicted_results) if len(predicted_results) > 0 else 0.0

    # Recall denominator = total gold rows
    recall = tp / len(gold_results) if len(gold_results) > 0 else 0.0

    return (float(precision), float(recall))


def compute_example_metrics(entry: dict) -> ExampleMetrics:
    """Build an ``ExampleMetrics`` for one evaluated entry."""
    question = entry.get("question", "")
    predicted_sparql = entry.get("predicted_sparql")
    gold_sparql = entry.get("gold_sparql")
    predicted_answer = entry.get("predicted_answer_tuple", [])
    gold_answer = entry.get("gold_answer_tuple", [])
    f1_val = entry.get("f1", 0.0)

    precision, recall = _precision_recall_from_cost_matrix(
        predicted_answer, gold_answer
    )

    is_em = abs(f1_val - 1.0) < 1e-9

    error_cat = categorize_error(
        predicted_sparql=predicted_sparql,
        predicted_answer=predicted_answer,
        gold_answer=gold_answer,
        f1_val=f1_val,
    )

    return ExampleMetrics(
        question=question,
        predicted_sparql=predicted_sparql,
        gold_sparql=gold_sparql,
        predicted_answer=predicted_answer,
        gold_answer=gold_answer,
        f1=f1_val,
        precision=precision,
        recall=recall,
        is_exact_match=is_em,
        error_category=error_cat,
        predicted_complexity=analyse_sparql_complexity(predicted_sparql),
        gold_complexity=analyse_sparql_complexity(gold_sparql),
    )


# ===================================================================
# 3. Error Categorization
# ===================================================================

# Human-readable taxonomy of failure modes
ERROR_CATEGORIES = OrderedDict(
    [
        ("correct", "F1 == 1 (exact match)"),
        ("high_partial", "0.5 <= F1 < 1 – close but not exact"),
        ("low_partial", "0 < F1 < 0.5 – partially correct"),
        ("empty_prediction", "Predicted SPARQL returned empty / None"),
        ("no_sparql", "No SPARQL was generated at all"),
        ("zero_f1", "F1 == 0 with non-empty prediction"),
    ]
)


def categorize_error(
    predicted_sparql: Optional[str],
    predicted_answer: Optional[list],
    gold_answer: Optional[list],
    f1_val: float,
) -> str:
    """Assign a human-readable error category."""
    if abs(f1_val - 1.0) < 1e-9:
        return "correct"
    if predicted_sparql is None:
        return "no_sparql"
    if not predicted_answer:
        return "empty_prediction"
    if f1_val >= 0.5:
        return "high_partial"
    if f1_val > 0:
        return "low_partial"
    return "zero_f1"


# ===================================================================
# 4. Aggregate Report
# ===================================================================

@dataclass
class AnalysisReport:
    """Full evaluation analysis report."""

    # ---- per-example ----
    examples: List[ExampleMetrics] = field(default_factory=list)

    # ---- aggregate ----
    total: int = 0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    exact_match_pct: float = 0.0
    none_pct: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_bucket_f1: Dict[str, float] = field(default_factory=dict)
    complexity_bucket_counts: Dict[str, int] = field(default_factory=dict)

    # ---- metadata ----
    source_file: str = ""

    # ----- builders -----
    @classmethod
    def from_entries(cls, entries: List[dict], source_file: str = "") -> "AnalysisReport":
        """Build a report from a list of evaluated entry dicts."""
        examples = [compute_example_metrics(e) for e in entries]
        report = cls(examples=examples, source_file=source_file)
        report._compute_aggregates()
        return report

    @classmethod
    def from_file(cls, filepath: str) -> "AnalysisReport":
        """Load an evaluated JSON and build the report."""
        with open(filepath, "r", encoding="utf-8") as fd:
            entries = json.load(fd)
        return cls.from_entries(entries, source_file=filepath)

    # ----- internals -----
    def _compute_aggregates(self):
        n = len(self.examples)
        self.total = n
        if n == 0:
            return

        self.macro_f1 = sum(e.f1 for e in self.examples) / n
        self.macro_precision = sum(e.precision for e in self.examples) / n
        self.macro_recall = sum(e.recall for e in self.examples) / n
        self.exact_match_pct = (
            sum(1 for e in self.examples if e.is_exact_match) / n * 100
        )
        self.none_pct = (
            sum(1 for e in self.examples if e.error_category in ("no_sparql", "empty_prediction"))
            / n
            * 100
        )

        # Error distribution
        cat_counter: Counter = Counter(e.error_category for e in self.examples)
        self.error_distribution = dict(cat_counter.most_common())

        # F1 bucketed by gold SPARQL complexity
        bucket_f1s: Dict[str, list] = defaultdict(list)
        for e in self.examples:
            bucket = (
                e.gold_complexity.complexity_bucket
                if e.gold_complexity
                else "unknown"
            )
            bucket_f1s[bucket].append(e.f1)

        self.complexity_bucket_f1 = {
            b: sum(vals) / len(vals) for b, vals in bucket_f1s.items()
        }
        self.complexity_bucket_counts = {
            b: len(vals) for b, vals in bucket_f1s.items()
        }

    # ----- display -----
    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            " SPINACH Evaluation Analysis Report",
            "=" * 60,
            f"  Source file     : {self.source_file}",
            f"  Total examples  : {self.total}",
            "",
            "  --- Aggregate Metrics ---",
            f"  Macro F1        : {self.macro_f1 * 100:.2f}%",
            f"  Macro Precision : {self.macro_precision * 100:.2f}%",
            f"  Macro Recall    : {self.macro_recall * 100:.2f}%",
            f"  Exact Match     : {self.exact_match_pct:.2f}%",
            f"  None answers    : {self.none_pct:.2f}%",
            "",
            "  --- Error Distribution ---",
        ]
        for cat, cnt in self.error_distribution.items():
            pct = cnt / self.total * 100 if self.total else 0
            desc = ERROR_CATEGORIES.get(cat, cat)
            lines.append(f"    {cat:20s} : {cnt:4d}  ({pct:5.1f}%)  – {desc}")

        lines.append("")
        lines.append("  --- F1 by Gold SPARQL Complexity ---")
        for bucket in ("simple", "medium", "complex", "very_complex", "unknown"):
            if bucket in self.complexity_bucket_f1:
                avg_f1 = self.complexity_bucket_f1[bucket]
                cnt = self.complexity_bucket_counts[bucket]
                lines.append(
                    f"    {bucket:15s} : F1 = {avg_f1 * 100:5.2f}%   (n={cnt})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable dict (no numpy / dataclass nesting)."""
        return {
            "source_file": self.source_file,
            "total": self.total,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "exact_match_pct": self.exact_match_pct,
            "none_pct": self.none_pct,
            "error_distribution": self.error_distribution,
            "complexity_bucket_f1": self.complexity_bucket_f1,
            "complexity_bucket_counts": self.complexity_bucket_counts,
            "examples": [
                {
                    "question": e.question,
                    "predicted_sparql": e.predicted_sparql,
                    "gold_sparql": e.gold_sparql,
                    "f1": e.f1,
                    "precision": e.precision,
                    "recall": e.recall,
                    "is_exact_match": e.is_exact_match,
                    "error_category": e.error_category,
                    "predicted_complexity_score": (
                        e.predicted_complexity.complexity_score
                        if e.predicted_complexity
                        else None
                    ),
                    "gold_complexity_score": (
                        e.gold_complexity.complexity_score
                        if e.gold_complexity
                        else None
                    ),
                    "gold_complexity_bucket": (
                        e.gold_complexity.complexity_bucket
                        if e.gold_complexity
                        else None
                    ),
                }
                for e in self.examples
            ],
        }

    def save_json(self, filepath: str):
        """Write the full report as JSON."""
        with open(filepath, "w", encoding="utf-8") as fd:
            json.dump(self.to_dict(), fd, indent=2, ensure_ascii=False)

    def save_csv(self, filepath: str):
        """Write per-example metrics as CSV."""
        if not self.examples:
            return
        fieldnames = [
            "question",
            "f1",
            "precision",
            "recall",
            "is_exact_match",
            "error_category",
            "predicted_sparql",
            "gold_sparql",
            "predicted_complexity_score",
            "gold_complexity_score",
            "gold_complexity_bucket",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as fd:
            writer = csv.DictWriter(fd, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.examples:
                writer.writerow(
                    {
                        "question": e.question,
                        "f1": f"{e.f1:.4f}",
                        "precision": f"{e.precision:.4f}",
                        "recall": f"{e.recall:.4f}",
                        "is_exact_match": e.is_exact_match,
                        "error_category": e.error_category,
                        "predicted_sparql": e.predicted_sparql or "",
                        "gold_sparql": e.gold_sparql or "",
                        "predicted_complexity_score": (
                            f"{e.predicted_complexity.complexity_score:.1f}"
                            if e.predicted_complexity
                            else ""
                        ),
                        "gold_complexity_score": (
                            f"{e.gold_complexity.complexity_score:.1f}"
                            if e.gold_complexity
                            else ""
                        ),
                        "gold_complexity_bucket": (
                            e.gold_complexity.complexity_bucket
                            if e.gold_complexity
                            else ""
                        ),
                    }
                )


# ===================================================================
# 5. Cross-Run Comparison / Benchmarking
# ===================================================================

@dataclass
class BenchmarkComparison:
    """Compare two or more evaluation runs side-by-side."""

    reports: List[AnalysisReport] = field(default_factory=list)

    def add(self, report: AnalysisReport):
        self.reports.append(report)

    def comparison_table(self) -> str:
        """Return a pretty-printed comparison table."""
        if not self.reports:
            return "No reports to compare."

        header = (
            f"{'Run':<40s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} "
            f"{'EM%':>7s} {'None%':>7s} {'N':>5s}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in self.reports:
            name = os.path.basename(r.source_file) if r.source_file else "unknown"
            lines.append(
                f"{name:<40s} "
                f"{r.macro_f1 * 100:6.2f}% "
                f"{r.macro_precision * 100:6.2f}% "
                f"{r.macro_recall * 100:6.2f}% "
                f"{r.exact_match_pct:6.2f}% "
                f"{r.none_pct:6.2f}% "
                f"{r.total:5d}"
            )
        lines.append(sep)

        # Per-complexity bucket comparison
        all_buckets = sorted(
            set(
                b
                for r in self.reports
                for b in r.complexity_bucket_f1.keys()
            )
        )
        if all_buckets:
            lines.append("")
            lines.append("F1 by Complexity Bucket:")
            bucket_header = f"{'Run':<40s} " + " ".join(
                f"{b:>14s}" for b in all_buckets
            )
            lines.append(bucket_header)
            for r in self.reports:
                name = (
                    os.path.basename(r.source_file) if r.source_file else "unknown"
                )
                vals = []
                for b in all_buckets:
                    v = r.complexity_bucket_f1.get(b)
                    vals.append(f"{v * 100:13.2f}%" if v is not None else f"{'N/A':>14s}")
                lines.append(f"{name:<40s} " + " ".join(vals))
            lines.append(sep)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "runs": [r.to_dict() for r in self.reports],
        }

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as fd:
            json.dump(self.to_dict(), fd, indent=2, ensure_ascii=False)


# ===================================================================
# 6. Convenience helpers
# ===================================================================

def analyse_file(filepath: str) -> AnalysisReport:
    """One-liner to get a full analysis report from an output JSON."""
    return AnalysisReport.from_file(filepath)


def compare_runs(*filepaths: str) -> BenchmarkComparison:
    """Build a BenchmarkComparison from a list of evaluated JSON files."""
    bench = BenchmarkComparison()
    for fp in filepaths:
        bench.add(AnalysisReport.from_file(fp))
    return bench


# ===================================================================
# CLI entry-point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SPINACH Evaluation Analysis & Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse a single run
  python spinach_agent/analysis.py --input output.json

  # Compare two runs
  python spinach_agent/analysis.py --input run1.json run2.json --compare

  # Export detailed CSV
  python spinach_agent/analysis.py --input output.json --output report.csv --format csv
""",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Path(s) to evaluated JSON output file(s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the analysis report (JSON or CSV).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple input files side-by-side.",
    )

    args = parser.parse_args()

    if args.compare and len(args.input) >= 2:
        bench = compare_runs(*args.input)
        print(bench.comparison_table())
        if args.output:
            bench.save_json(args.output)
            print(f"\nComparison report saved to {args.output}")
    else:
        for fp in args.input:
            report = analyse_file(fp)
            print(report.summary())
            if args.output:
                if args.format == "csv":
                    report.save_csv(args.output)
                else:
                    report.save_json(args.output)
                print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()

