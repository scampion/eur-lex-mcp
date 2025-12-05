#!/usr/bin/env python3
"""
EUR-Lex MCP Server Evaluation Runner.

This is the main orchestrator that runs the complete evaluation pipeline
following the six-phase workplan:

1. Corpus Acquisition and Stratified Sampling
2. Synthetic Query Generation
3. Hard Negative Mining and Pseudo-Label Generation
4. LLM-as-Judge Evaluation
5. Human Annotation Calibration (preparation)
6. Automated Metrics Computation and Coverage Testing

Usage:
    python run_evaluation.py --phase all          # Run all phases
    python run_evaluation.py --phase corpus       # Phase 1 only
    python run_evaluation.py --phase queries      # Phase 2 only
    python run_evaluation.py --phase labels       # Phase 3 only
    python run_evaluation.py --phase judge        # Phase 4 only
    python run_evaluation.py --phase annotation   # Phase 5 preparation
    python run_evaluation.py --phase metrics      # Phase 6 only
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    corpus_config,
    query_gen_config,
    pseudo_label_config,
    llm_judge_config,
    annotation_config,
    metrics_config,
    OUTPUT_DIR,
)
from utils import (
    setup_logging,
    create_run_log_dir,
    Timer,
    save_json,
    export_to_html_report,
)

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main evaluation orchestrator."""

    def __init__(
        self,
        run_dir: Optional[Path] = None,
        use_mock_client: bool = False,
    ):
        self.run_dir = run_dir or create_run_log_dir("eval")
        self.use_mock_client = use_mock_client
        self.results = {}

        # Setup logging to file
        log_file = self.run_dir / "evaluation.log"
        setup_logging(level=logging.INFO, log_file=log_file)

        logger.info(f"Evaluation run directory: {self.run_dir}")

    # ========================================================================
    # Phase 1: Corpus Acquisition
    # ========================================================================

    def run_phase1_corpus(self, targets: Optional[Dict[str, int]] = None) -> Dict:
        """Phase 1: Corpus acquisition and stratified sampling."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Corpus Acquisition and Stratified Sampling")
        logger.info("=" * 60)

        from corpus import CorpusManager

        with Timer("Phase 1"):
            manager = CorpusManager()
            corpus = manager.prepare_corpus(targets=targets, save=True)

            stats = {
                "total_documents": sum(len(docs) for docs in corpus.values()),
                "by_type": {
                    doc_type: len(docs) for doc_type, docs in corpus.items()
                },
            }

        self.results["phase1"] = stats
        logger.info(f"Phase 1 complete: {stats['total_documents']} documents")

        return stats

    # ========================================================================
    # Phase 2: Synthetic Query Generation
    # ========================================================================

    def run_phase2_queries(
        self,
        use_llm: bool = True,
        use_t5: bool = True,
    ) -> Dict:
        """Phase 2: Synthetic query generation."""
        logger.info("=" * 60)
        logger.info("PHASE 2: Synthetic Query Generation")
        logger.info("=" * 60)

        from corpus import CorpusManager
        from query_generation import QueryGenerationPipeline

        with Timer("Phase 2"):
            # Load corpus
            corpus_manager = CorpusManager()
            corpus = corpus_manager.load_corpus()

            # Generate queries
            pipeline = QueryGenerationPipeline()
            queries = pipeline.generate_all_queries(
                corpus,
                use_llm=use_llm,
                use_t5=use_t5,
            )

            # Save queries
            pipeline.save_queries(queries)

            # Compute statistics
            type_counts = {}
            method_counts = {}
            for q in queries:
                type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1
                method_counts[q.generation_method] = (
                    method_counts.get(q.generation_method, 0) + 1
                )

            stats = {
                "total_queries": len(queries),
                "by_type": type_counts,
                "by_method": method_counts,
            }

        self.results["phase2"] = stats
        logger.info(f"Phase 2 complete: {stats['total_queries']} queries generated")

        return stats

    # ========================================================================
    # Phase 3: Hard Negative Mining and Pseudo-Labels
    # ========================================================================

    async def run_phase3_labels(self) -> Dict:
        """Phase 3: Hard negative mining and pseudo-label generation."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Hard Negative Mining and Pseudo-Labels")
        logger.info("=" * 60)

        from corpus import CorpusManager
        from query_generation import QueryGenerationPipeline
        from pseudo_labels import PseudoLabelPipeline

        with Timer("Phase 3"):
            # Load corpus and queries
            corpus_manager = CorpusManager()
            corpus = corpus_manager.load_corpus()

            query_pipeline = QueryGenerationPipeline()
            queries = query_pipeline.load_queries()

            # Run pseudo-labeling
            pipeline = PseudoLabelPipeline()
            dataset_path = await pipeline.run(corpus, queries)

            stats = {
                "dataset_path": str(dataset_path),
                "corpus_size": len(corpus),
                "query_count": len(queries),
            }

        self.results["phase3"] = stats
        logger.info(f"Phase 3 complete: dataset saved to {dataset_path}")

        return stats

    # ========================================================================
    # Phase 4: LLM-as-Judge Evaluation
    # ========================================================================

    def run_phase4_judge(self, sample_size: int = 100) -> Dict:
        """Phase 4: LLM-as-judge evaluation."""
        logger.info("=" * 60)
        logger.info("PHASE 4: LLM-as-Judge Evaluation")
        logger.info("=" * 60)

        from corpus import CorpusManager
        from query_generation import QueryGenerationPipeline
        from llm_judge import LLMJudgePipeline

        with Timer("Phase 4"):
            # Load data
            corpus_manager = CorpusManager()
            corpus = corpus_manager.load_corpus()

            query_pipeline = QueryGenerationPipeline()
            queries = query_pipeline.load_queries()
            queries_dict = {q.query_id: q for q in queries}

            # Create sample pairs
            pairs = []
            for query in queries[:sample_size]:
                if query.source_celex in corpus:
                    pairs.append((query, corpus[query.source_celex]))

            # Run evaluation
            pipeline = LLMJudgePipeline()
            results = pipeline.evaluate_pairs(pairs, queries_dict, corpus)

            # Save and compute stats
            pipeline.save_results(results)
            stats = pipeline.compute_statistics(results)

        self.results["phase4"] = stats
        logger.info(f"Phase 4 complete: {stats['total_judgments']} judgments")

        return stats

    # ========================================================================
    # Phase 5: Human Annotation Preparation
    # ========================================================================

    def run_phase5_annotation(self) -> Dict:
        """Phase 5: Prepare annotation batch for human calibration."""
        logger.info("=" * 60)
        logger.info("PHASE 5: Human Annotation Preparation")
        logger.info("=" * 60)

        from corpus import CorpusManager
        from query_generation import QueryGenerationPipeline
        from annotation import AnnotationPipeline

        with Timer("Phase 5"):
            # Load data
            corpus_manager = CorpusManager()
            corpus = corpus_manager.load_corpus()

            query_pipeline = QueryGenerationPipeline()
            queries = query_pipeline.load_queries()

            # Prepare annotation batch
            pipeline = AnnotationPipeline()
            batch, export_path = pipeline.prepare_annotation_batch(
                queries=queries,
                documents=corpus,
                batch_id=f"calibration_{datetime.now().strftime('%Y%m%d')}",
            )

            stats = {
                "sample_count": len(batch.samples),
                "export_path": str(export_path),
            }

        self.results["phase5"] = stats
        logger.info(f"Phase 5 complete: {stats['sample_count']} samples exported")

        return stats

    # ========================================================================
    # Phase 6: Automated Metrics
    # ========================================================================

    async def run_phase6_metrics(
        self,
        run_coverage: bool = True,
        run_edge_cases: bool = True,
    ) -> Dict:
        """Phase 6: Automated metrics computation and coverage testing."""
        logger.info("=" * 60)
        logger.info("PHASE 6: Automated Metrics and Coverage Testing")
        logger.info("=" * 60)

        from pseudo_labels import BEIRDatasetBuilder
        from metrics import MetricsPipeline

        with Timer("Phase 6"):
            # Load qrels
            dataset_builder = BEIRDatasetBuilder()
            qrels = dataset_builder.load_qrels()

            # TODO: Load actual run results from MCP server
            # For now, we'll need to generate these from retrieval results
            run = {}  # Placeholder

            # Run evaluation
            pipeline = MetricsPipeline()
            results = await pipeline.run_full_evaluation(
                qrels=qrels,
                run=run,
                queries_by_type=None,  # Would be populated from actual queries
                log_to_mlflow=True,
            )

            # Save results
            output_path = self.run_dir / "evaluation_results.json"
            pipeline.save_results(results, output_path)

            # Generate HTML report
            report_path = self.run_dir / "evaluation_report.html"
            export_to_html_report(results, report_path)

        self.results["phase6"] = results
        pipeline.print_summary(results)

        return results

    # ========================================================================
    # Full Pipeline
    # ========================================================================

    async def run_all_phases(
        self,
        skip_llm: bool = False,
        sample_size: int = 100,
    ) -> Dict:
        """Run the complete evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING FULL EUR-LEX EVALUATION PIPELINE")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            # Phase 1: Corpus
            self.run_phase1_corpus()

            # Phase 2: Query Generation
            self.run_phase2_queries(use_llm=not skip_llm, use_t5=True)

            # Phase 3: Pseudo-labels
            await self.run_phase3_labels()

            # Phase 4: LLM Judge
            if not skip_llm:
                self.run_phase4_judge(sample_size=sample_size)

            # Phase 5: Annotation prep
            self.run_phase5_annotation()

            # Phase 6: Metrics
            await self.run_phase6_metrics()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.results["error"] = str(e)
            raise

        finally:
            end_time = datetime.now()
            self.results["duration_seconds"] = (end_time - start_time).total_seconds()
            self.results["completed_at"] = end_time.isoformat()

            # Save final results
            results_path = self.run_dir / "final_results.json"
            save_json(self.results, results_path)
            logger.info(f"Final results saved to {results_path}")

        return self.results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EUR-Lex MCP Server Evaluation Runner"
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "corpus", "queries", "labels", "judge", "annotation", "metrics"],
        help="Which phase to run (default: all)",
    )

    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based operations (useful for testing)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples for LLM judge evaluation (default: 100)",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock MCP client instead of real server",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create runner
    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = EvaluationRunner(
        run_dir=output_dir,
        use_mock_client=args.use_mock,
    )

    # Run requested phase(s)
    try:
        if args.phase == "all":
            results = await runner.run_all_phases(
                skip_llm=args.skip_llm,
                sample_size=args.sample_size,
            )
        elif args.phase == "corpus":
            results = runner.run_phase1_corpus()
        elif args.phase == "queries":
            results = runner.run_phase2_queries(
                use_llm=not args.skip_llm,
                use_t5=True,
            )
        elif args.phase == "labels":
            results = await runner.run_phase3_labels()
        elif args.phase == "judge":
            results = runner.run_phase4_judge(sample_size=args.sample_size)
        elif args.phase == "annotation":
            results = runner.run_phase5_annotation()
        elif args.phase == "metrics":
            results = await runner.run_phase6_metrics()

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {runner.run_dir}")
        print(json.dumps(results, indent=2, default=str))

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
