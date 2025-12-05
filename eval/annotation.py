"""
Phase 5: Minimal Human Annotation for Calibration.

This module handles:
- Stratified sampling for calibration samples
- Annotation interface preparation (Label Studio export)
- Inter-annotator agreement measurement
- Calibration of pseudo-labels and LLM judge
"""
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import annotation_config, OUTPUT_DIR
from corpus import Document
from query_generation import GeneratedQuery
from llm_judge import JudgmentResult, Confidence

logger = logging.getLogger(__name__)


# TREC-style relevance scale for legal documents
RELEVANCE_GUIDELINES = """
RELEVANCE SCALE (adapted from TREC):

3 - HIGHLY RELEVANT: Document directly addresses the legal query; contains
    binding or persuasive authority; would be essential for legal research

2 - RELEVANT: Contains useful legal information related to query; relevant
    dicta, related provisions, or background context

1 - MARGINALLY RELEVANT: Touches on topic but not directly applicable;
    different jurisdiction, superseded version, or tangential mention

0 - NOT RELEVANT: Off-topic; query terms appear without legal connection

SPECIAL CONSIDERATIONS:
- For case law queries: Distinguish holdings from dicta
- For regulatory queries: Check in-force status
- For procedural queries: Assess practical applicability
"""


@dataclass
class AnnotationSample:
    """A sample prepared for human annotation."""

    sample_id: str
    query_id: str
    query_text: str
    query_type: str
    doc_id: str
    doc_title: str
    doc_text: str
    doc_type: str

    # Pre-computed predictions for calibration comparison
    cross_encoder_score: Optional[float] = None
    llm_judgment: Optional[str] = None
    llm_confidence: Optional[str] = None

    # Human annotation fields
    human_relevance: Optional[int] = None  # 0-3 scale
    annotator_id: Optional[str] = None
    annotation_notes: Optional[str] = None


@dataclass
class AnnotationBatch:
    """A batch of samples for annotation."""

    batch_id: str
    samples: List[AnnotationSample]
    guidelines: str = RELEVANCE_GUIDELINES


class StratifiedSampler:
    """Selects samples for annotation using stratified sampling."""

    def __init__(self, config=annotation_config):
        self.config = config
        self.random = random.Random(42)

    def select_samples(
        self,
        queries: List[GeneratedQuery],
        documents: Dict[str, Document],
        judgments: Optional[List[JudgmentResult]] = None,
        pseudo_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[AnnotationSample]:
        """
        Select calibration samples using stratified sampling.

        Strategy:
        - 10-15 samples per major document type
        - Mix of high-confidence and low-confidence LLM judgments
        - Include edge cases (partial relevance, cross-references)
        """
        samples = []
        sample_id = 0

        # Group queries by document type
        queries_by_type = {}
        for query in queries:
            if query.source_celex in documents:
                doc = documents[query.source_celex]
                if doc.doc_type not in queries_by_type:
                    queries_by_type[doc.doc_type] = []
                queries_by_type[doc.doc_type].append(query)

        # Index judgments by query_id if available
        judgments_by_query = {}
        if judgments:
            for j in judgments:
                if j.query_id not in judgments_by_query:
                    judgments_by_query[j.query_id] = []
                judgments_by_query[j.query_id].append(j)

        # Sample from each document type
        for doc_type, type_queries in queries_by_type.items():
            n_samples = min(
                self.config.samples_per_doc_type,
                len(type_queries),
            )

            # Stratify by confidence if judgments available
            if judgments_by_query:
                sampled = self._stratify_by_confidence(
                    type_queries, judgments_by_query, n_samples
                )
            else:
                sampled = self.random.sample(type_queries, n_samples)

            for query in sampled:
                doc = documents[query.source_celex]

                # Get pre-computed scores
                ce_score = None
                llm_judge = None
                llm_conf = None

                if pseudo_scores and query.query_id in pseudo_scores:
                    ce_score = pseudo_scores[query.query_id].get(doc.celex)

                if query.query_id in judgments_by_query:
                    for j in judgments_by_query[query.query_id]:
                        if j.doc_id == doc.celex:
                            llm_judge = j.judgment
                            llm_conf = j.confidence
                            break

                samples.append(
                    AnnotationSample(
                        sample_id=f"sample_{sample_id:04d}",
                        query_id=query.query_id,
                        query_text=query.text,
                        query_type=query.query_type,
                        doc_id=doc.celex,
                        doc_title=doc.title,
                        doc_text=doc.text[:2000],  # Truncate for annotation
                        doc_type=doc.doc_type,
                        cross_encoder_score=ce_score,
                        llm_judgment=llm_judge,
                        llm_confidence=llm_conf,
                    )
                )
                sample_id += 1

        logger.info(f"Selected {len(samples)} samples for annotation")
        return samples

    def _stratify_by_confidence(
        self,
        queries: List[GeneratedQuery],
        judgments: Dict[str, List[JudgmentResult]],
        n_samples: int,
    ) -> List[GeneratedQuery]:
        """Stratify sampling by LLM judgment confidence."""
        low_conf = []
        high_conf = []

        for query in queries:
            if query.query_id in judgments:
                # Check confidence of any judgment for this query
                for j in judgments[query.query_id]:
                    if j.confidence == Confidence.LOW.value:
                        low_conf.append(query)
                        break
                else:
                    high_conf.append(query)
            else:
                # No judgment, treat as low confidence
                low_conf.append(query)

        # Sample mix: more low confidence for uncertainty
        n_low = min(len(low_conf), int(n_samples * 0.7))
        n_high = min(len(high_conf), n_samples - n_low)

        sampled = []
        if low_conf:
            sampled.extend(self.random.sample(low_conf, n_low))
        if high_conf:
            sampled.extend(self.random.sample(high_conf, n_high))

        return sampled


class LabelStudioExporter:
    """Exports annotation samples to Label Studio format."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR / "annotation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_batch(
        self,
        batch: AnnotationBatch,
        output_file: Optional[str] = None,
    ) -> Path:
        """Export batch to Label Studio JSON format."""
        output_file = output_file or f"{batch.batch_id}.json"
        output_path = self.output_dir / output_file

        tasks = []
        for sample in batch.samples:
            task = {
                "id": sample.sample_id,
                "data": {
                    "query": sample.query_text,
                    "query_type": sample.query_type,
                    "document_title": sample.doc_title,
                    "document_text": sample.doc_text,
                    "document_type": sample.doc_type,
                    "celex": sample.doc_id,
                    # Include predictions for reference
                    "cross_encoder_score": sample.cross_encoder_score,
                    "llm_judgment": sample.llm_judgment,
                },
                "predictions": [],
            }

            # Add LLM judgment as prediction if available
            if sample.llm_judgment:
                relevance_map = {"RELEVANT": 2, "NOT_RELEVANT": 0}
                task["predictions"].append({
                    "model_version": "llm_judge_v1",
                    "result": [{
                        "from_name": "relevance",
                        "to_name": "document",
                        "type": "rating",
                        "value": {"rating": relevance_map.get(sample.llm_judgment, 1)},
                    }],
                })

            tasks.append(task)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)

        logger.info(f"Exported {len(tasks)} tasks to {output_path}")
        return output_path

    def export_guidelines(self) -> Path:
        """Export annotation guidelines."""
        guidelines_path = self.output_dir / "guidelines.md"

        with open(guidelines_path, "w", encoding="utf-8") as f:
            f.write("# Annotation Guidelines\n\n")
            f.write(RELEVANCE_GUIDELINES)

        return guidelines_path


class InterAnnotatorAgreement:
    """Calculates inter-annotator agreement metrics."""

    def calculate_cohens_kappa(
        self,
        annotations1: List[int],
        annotations2: List[int],
    ) -> float:
        """
        Calculate Cohen's Kappa for inter-annotator agreement.

        Args:
            annotations1: First annotator's labels
            annotations2: Second annotator's labels

        Returns:
            Kappa score (-1 to 1, where >0.6 is substantial agreement)
        """
        try:
            from sklearn.metrics import cohen_kappa_score

            return cohen_kappa_score(annotations1, annotations2)
        except ImportError:
            logger.warning("sklearn not available, computing manually")
            return self._manual_kappa(annotations1, annotations2)

    def _manual_kappa(
        self, annotations1: List[int], annotations2: List[int]
    ) -> float:
        """Manual Cohen's Kappa calculation."""
        n = len(annotations1)
        if n == 0:
            return 0.0

        # Count agreements
        agreements = sum(1 for a, b in zip(annotations1, annotations2) if a == b)
        p_o = agreements / n  # Observed agreement

        # Count label frequencies
        from collections import Counter

        counts1 = Counter(annotations1)
        counts2 = Counter(annotations2)
        labels = set(counts1.keys()) | set(counts2.keys())

        # Expected agreement by chance
        p_e = sum(
            (counts1.get(label, 0) / n) * (counts2.get(label, 0) / n)
            for label in labels
        )

        if p_e == 1:
            return 1.0

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def calculate_agreement_report(
        self,
        samples: List[AnnotationSample],
        annotator1_id: str,
        annotator2_id: str,
    ) -> Dict:
        """
        Generate agreement report for overlapping annotations.

        Returns:
            Dict with agreement metrics and disagreement analysis
        """
        # Extract overlapping annotations
        ann1 = []
        ann2 = []
        disagreements = []

        for sample in samples:
            # This assumes annotations are stored with annotator IDs
            # In practice, you'd load from separate annotation files
            if sample.annotator_id == annotator1_id:
                ann1.append(sample.human_relevance)
            elif sample.annotator_id == annotator2_id:
                ann2.append(sample.human_relevance)

        if len(ann1) != len(ann2):
            logger.warning("Unequal annotation counts")
            min_len = min(len(ann1), len(ann2))
            ann1 = ann1[:min_len]
            ann2 = ann2[:min_len]

        kappa = self.calculate_cohens_kappa(ann1, ann2)

        # Identify disagreements
        for i, (a, b) in enumerate(zip(ann1, ann2)):
            if a != b:
                disagreements.append({
                    "index": i,
                    "annotator1": a,
                    "annotator2": b,
                    "diff": abs(a - b),
                })

        return {
            "cohens_kappa": kappa,
            "total_samples": len(ann1),
            "agreement_count": len(ann1) - len(disagreements),
            "disagreement_count": len(disagreements),
            "agreement_rate": (len(ann1) - len(disagreements)) / len(ann1) if ann1 else 0,
            "target_kappa": annotation_config.target_kappa,
            "meets_target": kappa >= annotation_config.target_kappa,
            "disagreements": disagreements[:10],  # Sample of disagreements
        }


class CalibrationAnalyzer:
    """Analyzes calibration between automated and human labels."""

    def __init__(self, config=annotation_config):
        self.config = config

    def compare_cross_encoder(
        self,
        samples: List[AnnotationSample],
    ) -> Dict:
        """Compare cross-encoder pseudo-labels to human labels."""
        pairs = []
        for sample in samples:
            if sample.human_relevance is not None and sample.cross_encoder_score is not None:
                pairs.append((sample.cross_encoder_score, sample.human_relevance))

        if not pairs:
            return {"error": "No valid pairs for comparison"}

        # Analyze threshold calibration
        thresholds = [0.2, 0.5, 0.8]
        results = {}

        for thresh in thresholds:
            predicted = [1 if score >= thresh else 0 for score, _ in pairs]
            actual = [1 if label >= 2 else 0 for _, label in pairs]

            correct = sum(1 for p, a in zip(predicted, actual) if p == a)
            accuracy = correct / len(pairs)
            results[f"threshold_{thresh}"] = {
                "accuracy": accuracy,
                "n_samples": len(pairs),
            }

        return results

    def compare_llm_judge(
        self,
        samples: List[AnnotationSample],
    ) -> Dict:
        """Compare LLM judge outputs to human labels."""
        pairs = []
        for sample in samples:
            if sample.human_relevance is not None and sample.llm_judgment is not None:
                # Convert LLM binary to comparable format
                llm_binary = 1 if sample.llm_judgment == "RELEVANT" else 0
                human_binary = 1 if sample.human_relevance >= 2 else 0
                pairs.append((llm_binary, human_binary, sample.llm_confidence))

        if not pairs:
            return {"error": "No valid pairs for comparison"}

        # Overall agreement
        correct = sum(1 for llm, human, _ in pairs if llm == human)
        agreement_rate = correct / len(pairs)

        # Agreement by confidence level
        by_confidence = {}
        for llm, human, conf in pairs:
            if conf not in by_confidence:
                by_confidence[conf] = {"correct": 0, "total": 0}
            by_confidence[conf]["total"] += 1
            if llm == human:
                by_confidence[conf]["correct"] += 1

        for conf in by_confidence:
            by_confidence[conf]["accuracy"] = (
                by_confidence[conf]["correct"] / by_confidence[conf]["total"]
            )

        return {
            "overall_agreement": agreement_rate,
            "total_samples": len(pairs),
            "meets_target": agreement_rate >= self.config.llm_human_agreement_target,
            "target": self.config.llm_human_agreement_target,
            "by_confidence": by_confidence,
        }

    def suggest_threshold_adjustments(
        self,
        cross_encoder_results: Dict,
        llm_results: Dict,
    ) -> List[str]:
        """Suggest adjustments based on calibration analysis."""
        suggestions = []

        # Cross-encoder threshold suggestions
        best_thresh = max(
            [k for k in cross_encoder_results if k.startswith("threshold_")],
            key=lambda k: cross_encoder_results[k]["accuracy"],
            default=None,
        )
        if best_thresh:
            suggestions.append(
                f"Best cross-encoder threshold: {best_thresh.split('_')[1]} "
                f"(accuracy: {cross_encoder_results[best_thresh]['accuracy']:.2%})"
            )

        # LLM judge suggestions
        if "overall_agreement" in llm_results:
            if llm_results["overall_agreement"] < self.config.llm_human_agreement_target:
                suggestions.append(
                    f"LLM agreement ({llm_results['overall_agreement']:.2%}) below target. "
                    "Consider prompt refinement."
                )

            # Check if high-confidence predictions are reliable
            by_conf = llm_results.get("by_confidence", {})
            if "HIGH" in by_conf and by_conf["HIGH"]["accuracy"] < 0.9:
                suggestions.append(
                    "High-confidence predictions have low accuracy. "
                    "Review confidence calibration."
                )

        return suggestions


class AnnotationPipeline:
    """Main pipeline for human annotation workflow."""

    def __init__(self, config=annotation_config):
        self.config = config
        self.sampler = StratifiedSampler(config)
        self.exporter = LabelStudioExporter()
        self.agreement = InterAnnotatorAgreement()
        self.calibration = CalibrationAnalyzer(config)

    def prepare_annotation_batch(
        self,
        queries: List[GeneratedQuery],
        documents: Dict[str, Document],
        judgments: Optional[List[JudgmentResult]] = None,
        batch_id: str = "batch_001",
    ) -> Tuple[AnnotationBatch, Path]:
        """Prepare and export annotation batch."""
        # Select samples
        samples = self.sampler.select_samples(queries, documents, judgments)

        # Create batch
        batch = AnnotationBatch(
            batch_id=batch_id,
            samples=samples,
            guidelines=RELEVANCE_GUIDELINES,
        )

        # Export to Label Studio
        export_path = self.exporter.export_batch(batch)
        self.exporter.export_guidelines()

        return batch, export_path

    def load_annotations(self, annotations_file: Path) -> List[AnnotationSample]:
        """Load completed annotations from Label Studio export."""
        with open(annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            # Extract annotation result
            annotations = item.get("annotations", [])
            relevance = None
            annotator = None

            if annotations:
                for ann in annotations[0].get("result", []):
                    if ann.get("from_name") == "relevance":
                        relevance = ann["value"]["rating"]
                annotator = annotations[0].get("completed_by")

            samples.append(
                AnnotationSample(
                    sample_id=item["id"],
                    query_id=item["data"].get("query_id", ""),
                    query_text=item["data"]["query"],
                    query_type=item["data"]["query_type"],
                    doc_id=item["data"]["celex"],
                    doc_title=item["data"]["document_title"],
                    doc_text=item["data"]["document_text"],
                    doc_type=item["data"]["document_type"],
                    cross_encoder_score=item["data"].get("cross_encoder_score"),
                    llm_judgment=item["data"].get("llm_judgment"),
                    human_relevance=relevance,
                    annotator_id=annotator,
                )
            )

        return samples

    def run_calibration(
        self,
        annotated_samples: List[AnnotationSample],
    ) -> Dict:
        """Run full calibration analysis."""
        results = {
            "cross_encoder": self.calibration.compare_cross_encoder(annotated_samples),
            "llm_judge": self.calibration.compare_llm_judge(annotated_samples),
        }

        results["suggestions"] = self.calibration.suggest_threshold_adjustments(
            results["cross_encoder"],
            results["llm_judge"],
        )

        return results


def main():
    """Main entry point for annotation preparation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from corpus import CorpusManager
    from query_generation import QueryGenerationPipeline

    # Load data
    logger.info("Loading corpus and queries...")
    corpus_manager = CorpusManager()
    corpus = corpus_manager.load_corpus()

    query_pipeline = QueryGenerationPipeline()
    queries = query_pipeline.load_queries()

    # Prepare annotation batch
    pipeline = AnnotationPipeline()
    batch, export_path = pipeline.prepare_annotation_batch(
        queries=queries,
        documents=corpus,
        batch_id="calibration_batch_001",
    )

    print(f"\nAnnotation preparation complete!")
    print(f"Samples: {len(batch.samples)}")
    print(f"Exported to: {export_path}")
    print("\nNext steps:")
    print("1. Import into Label Studio")
    print("2. Complete annotations (targeting 50-100 samples)")
    print("3. Export results and run calibration analysis")


if __name__ == "__main__":
    main()
