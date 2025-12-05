"""
Phase 6: Automated Metrics Computation and Coverage Testing.

This module implements:
- Core retrieval metrics (nDCG, MRR, Precision, Recall, MAP)
- Coverage testing across EUR-Lex document types
- Edge case test suite
- MLflow experiment tracking
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from config import metrics_config, OUTPUT_DIR

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    ndcg_5: float
    ndcg_10: float
    mrr: float
    precision_5: float
    precision_10: float
    recall_10: float
    recall_100: float
    map_100: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def meets_success_criteria(self, config=metrics_config) -> Dict[str, bool]:
        """Check if metrics meet success criteria."""
        return {
            "ndcg_10": self.ndcg_10 >= config.target_ndcg_10,
            "mrr": self.mrr >= config.target_mrr,
        }


@dataclass
class CoverageTestResult:
    """Result of a coverage test."""

    test_id: str
    query: str
    doc_type: str
    expected_celex: Optional[str]
    returned_results: List[str]
    passed: bool
    error: Optional[str] = None


@dataclass
class EdgeCaseResult:
    """Result of an edge case test."""

    test_id: str
    query: str
    category: str
    must_return: Optional[str]
    top_results: List[str]
    passed: bool
    notes: str = ""


class MetricsCalculator:
    """Calculates standard IR metrics using ranx library."""

    def __init__(self):
        self._ranx_available = None

    @property
    def ranx_available(self) -> bool:
        """Check if ranx is available."""
        if self._ranx_available is None:
            try:
                import ranx

                self._ranx_available = True
            except ImportError:
                self._ranx_available = False
        return self._ranx_available

    def calculate_metrics(
        self,
        qrels: Dict[str, Dict[str, int]],
        run: Dict[str, Dict[str, float]],
    ) -> RetrievalMetrics:
        """
        Calculate retrieval metrics.

        Args:
            qrels: Ground truth relevance labels
                   {query_id: {doc_id: relevance_score}}
            run: Retrieval results with scores
                 {query_id: {doc_id: retrieval_score}}

        Returns:
            RetrievalMetrics object with all computed metrics
        """
        if self.ranx_available:
            return self._calculate_with_ranx(qrels, run)
        else:
            return self._calculate_manually(qrels, run)

    def _calculate_with_ranx(
        self,
        qrels: Dict[str, Dict[str, int]],
        run: Dict[str, Dict[str, float]],
    ) -> RetrievalMetrics:
        """Calculate metrics using ranx library."""
        from ranx import Qrels, Run, evaluate

        ranx_qrels = Qrels(qrels)
        ranx_run = Run(run)

        metrics = evaluate(
            ranx_qrels,
            ranx_run,
            [
                "ndcg@5",
                "ndcg@10",
                "mrr",
                "precision@5",
                "precision@10",
                "recall@10",
                "recall@100",
                "map@100",
            ],
        )

        return RetrievalMetrics(
            ndcg_5=metrics["ndcg@5"],
            ndcg_10=metrics["ndcg@10"],
            mrr=metrics["mrr"],
            precision_5=metrics["precision@5"],
            precision_10=metrics["precision@10"],
            recall_10=metrics["recall@10"],
            recall_100=metrics["recall@100"],
            map_100=metrics["map@100"],
        )

    def _calculate_manually(
        self,
        qrels: Dict[str, Dict[str, int]],
        run: Dict[str, Dict[str, float]],
    ) -> RetrievalMetrics:
        """Calculate metrics without ranx (fallback implementation)."""
        metrics = {
            "ndcg@5": [],
            "ndcg@10": [],
            "mrr": [],
            "precision@5": [],
            "precision@10": [],
            "recall@10": [],
            "recall@100": [],
            "map@100": [],
        }

        for query_id, relevances in qrels.items():
            if query_id not in run:
                continue

            # Sort run by score
            ranked = sorted(run[query_id].items(), key=lambda x: -x[1])
            ranked_ids = [doc_id for doc_id, _ in ranked]

            # Calculate metrics for this query
            metrics["ndcg@5"].append(self._ndcg_at_k(ranked_ids, relevances, 5))
            metrics["ndcg@10"].append(self._ndcg_at_k(ranked_ids, relevances, 10))
            metrics["mrr"].append(self._mrr(ranked_ids, relevances))
            metrics["precision@5"].append(self._precision_at_k(ranked_ids, relevances, 5))
            metrics["precision@10"].append(self._precision_at_k(ranked_ids, relevances, 10))
            metrics["recall@10"].append(self._recall_at_k(ranked_ids, relevances, 10))
            metrics["recall@100"].append(self._recall_at_k(ranked_ids, relevances, 100))
            metrics["map@100"].append(self._average_precision(ranked_ids, relevances, 100))

        # Average across queries
        return RetrievalMetrics(
            ndcg_5=sum(metrics["ndcg@5"]) / len(metrics["ndcg@5"]) if metrics["ndcg@5"] else 0,
            ndcg_10=sum(metrics["ndcg@10"]) / len(metrics["ndcg@10"]) if metrics["ndcg@10"] else 0,
            mrr=sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0,
            precision_5=sum(metrics["precision@5"]) / len(metrics["precision@5"]) if metrics["precision@5"] else 0,
            precision_10=sum(metrics["precision@10"]) / len(metrics["precision@10"]) if metrics["precision@10"] else 0,
            recall_10=sum(metrics["recall@10"]) / len(metrics["recall@10"]) if metrics["recall@10"] else 0,
            recall_100=sum(metrics["recall@100"]) / len(metrics["recall@100"]) if metrics["recall@100"] else 0,
            map_100=sum(metrics["map@100"]) / len(metrics["map@100"]) if metrics["map@100"] else 0,
        )

    def _ndcg_at_k(
        self, ranked: List[str], relevances: Dict[str, int], k: int
    ) -> float:
        """Calculate nDCG@k."""
        import math

        dcg = 0.0
        for i, doc_id in enumerate(ranked[:k]):
            rel = relevances.get(doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = sorted(relevances.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def _mrr(self, ranked: List[str], relevances: Dict[str, int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(ranked):
            if relevances.get(doc_id, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def _precision_at_k(
        self, ranked: List[str], relevances: Dict[str, int], k: int
    ) -> float:
        """Calculate Precision@k."""
        relevant = sum(1 for doc_id in ranked[:k] if relevances.get(doc_id, 0) > 0)
        return relevant / k

    def _recall_at_k(
        self, ranked: List[str], relevances: Dict[str, int], k: int
    ) -> float:
        """Calculate Recall@k."""
        total_relevant = sum(1 for rel in relevances.values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        retrieved_relevant = sum(
            1 for doc_id in ranked[:k] if relevances.get(doc_id, 0) > 0
        )
        return retrieved_relevant / total_relevant

    def _average_precision(
        self, ranked: List[str], relevances: Dict[str, int], k: int
    ) -> float:
        """Calculate Average Precision."""
        precisions = []
        relevant_count = 0
        for i, doc_id in enumerate(ranked[:k]):
            if relevances.get(doc_id, 0) > 0:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        total_relevant = sum(1 for rel in relevances.values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        return sum(precisions) / total_relevant


class CoverageTester:
    """Tests coverage across EUR-Lex document types."""

    # Coverage test matrix from workplan
    COVERAGE_MATRIX = {
        "regulation": {"celex_sector": "3R", "min_queries": 100},
        "directive": {"celex_sector": "3L", "min_queries": 100},
        "decision": {"celex_sector": "3D", "min_queries": 80},
        "caselaw_ecj": {"celex_sector": "6CJ", "min_queries": 100},
        "caselaw_gc": {"celex_sector": "6TJ", "min_queries": 60},
        "ag_opinions": {"celex_sector": "6CC", "min_queries": 40},
        "proposal": {"celex_sector": "5PC", "min_queries": 60},
        "intagr": {"celex_sector": "2A", "min_queries": 40},
        "consolidated": {"celex_sector": "0", "min_queries": 40},
    }

    def __init__(self, mcp_client=None):
        self._mcp_client = mcp_client

    @property
    def mcp_client(self):
        """Lazy load MCP client."""
        if self._mcp_client is None:
            from mcp_client import MCPClient

            self._mcp_client = MCPClient()
        return self._mcp_client

    async def run_coverage_tests(
        self,
        queries_by_type: Dict[str, List[Tuple[str, str]]],  # {doc_type: [(query_id, query_text)]}
    ) -> Dict[str, List[CoverageTestResult]]:
        """Run coverage tests for all document types."""
        results = {}

        for doc_type, config in self.COVERAGE_MATRIX.items():
            if doc_type not in queries_by_type:
                logger.warning(f"No queries for document type: {doc_type}")
                continue

            type_queries = queries_by_type[doc_type]
            type_results = []

            for query_id, query_text in tqdm(
                type_queries[: config["min_queries"]],
                desc=f"Testing {doc_type}",
            ):
                try:
                    search_results = await self.mcp_client.search(query_text, limit=10)
                    returned = [r["celex"] for r in search_results]

                    # Check if any returned result matches the expected sector
                    sector = config["celex_sector"]
                    has_correct_type = any(
                        celex.startswith(sector) or sector in celex
                        for celex in returned
                    )

                    type_results.append(
                        CoverageTestResult(
                            test_id=f"{doc_type}_{query_id}",
                            query=query_text,
                            doc_type=doc_type,
                            expected_celex=None,
                            returned_results=returned[:5],
                            passed=has_correct_type,
                        )
                    )
                except Exception as e:
                    type_results.append(
                        CoverageTestResult(
                            test_id=f"{doc_type}_{query_id}",
                            query=query_text,
                            doc_type=doc_type,
                            expected_celex=None,
                            returned_results=[],
                            passed=False,
                            error=str(e),
                        )
                    )

            results[doc_type] = type_results

        return results

    def compute_coverage_stats(
        self, results: Dict[str, List[CoverageTestResult]]
    ) -> Dict:
        """Compute coverage statistics."""
        stats = {
            "by_type": {},
            "overall_pass_rate": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
        }

        for doc_type, type_results in results.items():
            passed = sum(1 for r in type_results if r.passed)
            total = len(type_results)

            stats["by_type"][doc_type] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
            }

            stats["total_tests"] += total
            stats["passed_tests"] += passed

        stats["overall_pass_rate"] = (
            stats["passed_tests"] / stats["total_tests"]
            if stats["total_tests"] > 0
            else 0.0
        )

        return stats


class EdgeCaseTester:
    """Tests edge cases and known challenges."""

    # Edge case test suite from workplan
    EDGE_CASES = [
        # Citation format variations
        {"query": "Regulation (EU) 2016/679", "must_return": "32016R0679", "category": "citation"},
        {"query": "GDPR", "must_return": "32016R0679", "category": "acronym"},
        {"query": "Case C-131/12", "must_return": "62012CJ0131", "category": "case_citation"},
        {"query": "Directive 95/46/EC", "must_return": "31995L0046", "category": "citation"},
        # Special characters
        {"query": "Article 101(1) TFEU", "category": "parentheses"},
        {"query": "Section 230", "category": "numeric_section"},
        # Boolean queries
        {"query": "data protection AND portability", "category": "boolean"},
        {"query": "GDPR OR data protection directive", "category": "boolean"},
        # Multilingual (if supported)
        {"query": "Datenschutz-Grundverordnung", "must_return": "32016R0679", "category": "multilingual"},
        {"query": "Règlement général sur la protection des données", "must_return": "32016R0679", "category": "multilingual"},
        # Temporal queries
        {"query": "directive 2006/54 amendments", "category": "temporal"},
        {"query": "regulation 2019/1150 consolidated", "category": "temporal"},
        # Abbreviations
        {"query": "ECJ jurisdiction", "category": "abbreviation"},
        {"query": "CJEU preliminary ruling", "category": "abbreviation"},
    ]

    def __init__(self, mcp_client=None):
        self._mcp_client = mcp_client

    @property
    def mcp_client(self):
        """Lazy load MCP client."""
        if self._mcp_client is None:
            from mcp_client import MCPClient

            self._mcp_client = MCPClient()
        return self._mcp_client

    async def run_edge_case_tests(self) -> List[EdgeCaseResult]:
        """Run all edge case tests."""
        results = []

        for i, test_case in enumerate(tqdm(self.EDGE_CASES, desc="Edge case tests")):
            query = test_case["query"]
            must_return = test_case.get("must_return")
            category = test_case["category"]

            try:
                search_results = await self.mcp_client.search(query, limit=10)
                top_results = [r["celex"] for r in search_results]

                # Check if must_return document is in results
                if must_return:
                    passed = must_return in top_results
                    notes = f"Expected {must_return}, got: {top_results[:3]}"
                else:
                    # For tests without must_return, just check we got results
                    passed = len(top_results) > 0
                    notes = f"Returned {len(top_results)} results"

                results.append(
                    EdgeCaseResult(
                        test_id=f"edge_{i:03d}",
                        query=query,
                        category=category,
                        must_return=must_return,
                        top_results=top_results[:5],
                        passed=passed,
                        notes=notes,
                    )
                )
            except Exception as e:
                results.append(
                    EdgeCaseResult(
                        test_id=f"edge_{i:03d}",
                        query=query,
                        category=category,
                        must_return=must_return,
                        top_results=[],
                        passed=False,
                        notes=f"Error: {str(e)}",
                    )
                )

        return results

    def compute_edge_case_stats(self, results: List[EdgeCaseResult]) -> Dict:
        """Compute edge case statistics."""
        by_category = {}
        for result in results:
            if result.category not in by_category:
                by_category[result.category] = {"passed": 0, "total": 0}
            by_category[result.category]["total"] += 1
            if result.passed:
                by_category[result.category]["passed"] += 1

        for cat in by_category:
            by_category[cat]["pass_rate"] = (
                by_category[cat]["passed"] / by_category[cat]["total"]
            )

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        return {
            "overall_pass_rate": passed / total if total > 0 else 0.0,
            "total_tests": total,
            "passed_tests": passed,
            "by_category": by_category,
            "failed_tests": [asdict(r) for r in results if not r.passed],
        }


class MLflowLogger:
    """MLflow experiment tracking integration."""

    def __init__(self, config=metrics_config):
        self.config = config
        self._mlflow_available = None

    @property
    def mlflow_available(self) -> bool:
        """Check if MLflow is available."""
        if self._mlflow_available is None:
            try:
                import mlflow

                self._mlflow_available = True
            except ImportError:
                self._mlflow_available = False
        return self._mlflow_available

    def log_evaluation_run(
        self,
        metrics: RetrievalMetrics,
        coverage_stats: Dict,
        edge_case_stats: Dict,
        run_name: str = "evaluation",
        params: Optional[Dict] = None,
    ) -> Optional[str]:
        """Log evaluation run to MLflow."""
        if not self.mlflow_available:
            logger.warning("MLflow not available, skipping logging")
            return None

        import mlflow

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"{run_name}_{datetime.now().isoformat()}"):
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log retrieval metrics
            mlflow.log_metrics(metrics.to_dict())

            # Log coverage stats
            mlflow.log_metric("coverage_pass_rate", coverage_stats["overall_pass_rate"])
            mlflow.log_metric("coverage_total_tests", coverage_stats["total_tests"])

            # Log edge case stats
            mlflow.log_metric("edge_case_pass_rate", edge_case_stats["overall_pass_rate"])
            mlflow.log_metric("edge_case_total_tests", edge_case_stats["total_tests"])

            # Log success criteria
            success = metrics.meets_success_criteria()
            for criterion, met in success.items():
                mlflow.log_metric(f"success_{criterion}", int(met))

            # Log artifacts
            results_file = OUTPUT_DIR / "evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "metrics": metrics.to_dict(),
                        "coverage": coverage_stats,
                        "edge_cases": edge_case_stats,
                        "success_criteria": success,
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact(str(results_file))

            return mlflow.active_run().info.run_id


class MetricsPipeline:
    """Main pipeline for metrics computation and evaluation."""

    def __init__(self, config=metrics_config):
        self.config = config
        self.calculator = MetricsCalculator()
        self.coverage_tester = CoverageTester()
        self.edge_case_tester = EdgeCaseTester()
        self.mlflow_logger = MLflowLogger(config)

    async def run_full_evaluation(
        self,
        qrels: Dict[str, Dict[str, int]],
        run: Dict[str, Dict[str, float]],
        queries_by_type: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        log_to_mlflow: bool = True,
    ) -> Dict:
        """Run complete evaluation pipeline."""
        results = {}

        # Core retrieval metrics
        logger.info("Computing retrieval metrics...")
        metrics = self.calculator.calculate_metrics(qrels, run)
        results["metrics"] = metrics.to_dict()
        results["success_criteria"] = metrics.meets_success_criteria()

        # Coverage testing
        if queries_by_type:
            logger.info("Running coverage tests...")
            coverage_results = await self.coverage_tester.run_coverage_tests(queries_by_type)
            results["coverage"] = self.coverage_tester.compute_coverage_stats(coverage_results)

        # Edge case testing
        logger.info("Running edge case tests...")
        edge_results = await self.edge_case_tester.run_edge_case_tests()
        results["edge_cases"] = self.edge_case_tester.compute_edge_case_stats(edge_results)

        # Log to MLflow
        if log_to_mlflow and self.mlflow_logger.mlflow_available:
            run_id = self.mlflow_logger.log_evaluation_run(
                metrics=metrics,
                coverage_stats=results.get("coverage", {"overall_pass_rate": 0, "total_tests": 0}),
                edge_case_stats=results["edge_cases"],
            )
            results["mlflow_run_id"] = run_id

        return results

    def save_results(
        self, results: Dict, output_path: Optional[Path] = None
    ) -> Path:
        """Save evaluation results."""
        output_path = output_path or OUTPUT_DIR / "evaluation_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")
        return output_path

    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\n--- Retrieval Metrics ---")
        for metric, value in results.get("metrics", {}).items():
            print(f"  {metric}: {value:.4f}")

        print("\n--- Success Criteria ---")
        for criterion, met in results.get("success_criteria", {}).items():
            status = "PASS" if met else "FAIL"
            print(f"  {criterion}: {status}")

        if "coverage" in results:
            print("\n--- Coverage ---")
            print(f"  Overall pass rate: {results['coverage']['overall_pass_rate']:.2%}")
            print(f"  Total tests: {results['coverage']['total_tests']}")

        if "edge_cases" in results:
            print("\n--- Edge Cases ---")
            print(f"  Overall pass rate: {results['edge_cases']['overall_pass_rate']:.2%}")
            print(f"  Total tests: {results['edge_cases']['total_tests']}")
            if results["edge_cases"].get("failed_tests"):
                print(f"  Failed tests: {len(results['edge_cases']['failed_tests'])}")

        print("\n" + "=" * 60)


async def main():
    """Main entry point for metrics computation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from pseudo_labels import BEIRDatasetBuilder

    # Load qrels from BEIR dataset
    logger.info("Loading evaluation data...")
    dataset_builder = BEIRDatasetBuilder()
    qrels = dataset_builder.load_qrels()

    # TODO: Load run results from MCP server evaluation
    # For now, create dummy run for demonstration
    run = {}  # Would be populated from actual retrieval results

    # Run evaluation
    pipeline = MetricsPipeline()
    results = await pipeline.run_full_evaluation(
        qrels=qrels,
        run=run,
        log_to_mlflow=True,
    )

    # Save and display
    pipeline.save_results(results)
    pipeline.print_summary(results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
