"""
Shared Utilities for EUR-Lex Evaluation.

This module provides common utilities used across the evaluation pipeline.
"""
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Tuple, Union

from config import CACHE_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Caching Utilities
# ============================================================================


class DiskCache:
    """Simple disk-based cache for expensive computations."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        # Check TTL
        mtime = cache_path.stat().st_mtime
        if time.time() - mtime > self.ttl_seconds:
            cache_path.unlink()
            return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cached values."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def cached(cache: DiskCache, key_fn: Callable[..., str]):
    """Decorator for caching function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = key_fn(*args, **kwargs)
            cached_value = cache.get(key)

            if cached_value is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_value

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator


# ============================================================================
# Text Processing Utilities
# ============================================================================


def truncate_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """Truncate text to maximum characters."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove control characters
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text.strip()


def tokenize_simple(text: str) -> List[str]:
    """Simple word tokenization."""
    import re

    text = text.lower()
    # Split on non-alphanumeric characters
    tokens = re.split(r"[^a-z0-9]+", text)
    # Filter empty and very short tokens
    return [t for t in tokens if len(t) > 1]


def compute_text_hash(text: str) -> str:
    """Compute hash of text for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ============================================================================
# CELEX Utilities
# ============================================================================


def parse_celex(celex: str) -> Dict[str, str]:
    """
    Parse CELEX number into components.

    CELEX format: [Sector][Year][Type][Number]
    Example: 32016R0679 -> Sector 3, Year 2016, Type R (Regulation), Number 0679
    """
    if not celex or len(celex) < 5:
        return {"error": "Invalid CELEX"}

    return {
        "sector": celex[0],
        "year": celex[1:5],
        "type": celex[5] if len(celex) > 5 else "",
        "number": celex[6:] if len(celex) > 6 else "",
        "full": celex,
    }


def celex_to_sector_name(sector: str) -> str:
    """Map CELEX sector number to name."""
    sectors = {
        "0": "Consolidated texts",
        "1": "Treaties",
        "2": "International agreements",
        "3": "Legislation",
        "4": "Complementary legislation",
        "5": "Preparatory acts",
        "6": "Case law",
        "7": "National implementation",
        "8": "References to national case law",
        "9": "Parliamentary questions",
        "C": "EFTA documents",
        "E": "EFTA documents",
    }
    return sectors.get(sector, "Unknown")


def celex_to_doc_type(type_code: str) -> str:
    """Map CELEX type code to document type."""
    types = {
        "R": "Regulation",
        "L": "Directive",
        "D": "Decision",
        "H": "Recommendation",
        "A": "Opinion",
        "CJ": "ECJ Judgment",
        "TJ": "General Court Judgment",
        "CC": "AG Opinion",
        "PC": "COM Proposal",
        "DC": "Council Working Document",
    }
    return types.get(type_code, "Unknown")


# ============================================================================
# Progress and Timing Utilities
# ============================================================================


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.info(f"{self.name}: {self.elapsed:.2f}s")


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n

    def get_stats(self) -> Dict:
        """Get progress statistics."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        return {
            "current": self.current,
            "total": self.total,
            "percent": self.current / self.total * 100 if self.total > 0 else 0,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta,
        }


# ============================================================================
# File I/O Utilities
# ============================================================================


def save_jsonl(data: List[Dict], path: Path):
    """Save list of dicts to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file to list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Batch Processing Utilities
# ============================================================================


def batch_iterator(items: List[T], batch_size: int):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    n_workers: int = 4,
    desc: str = "Processing",
) -> List[Any]:
    """Parallel map with progress bar."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(items),
            desc=desc,
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                results[idx] = None

    return results


# ============================================================================
# Logging Utilities
# ============================================================================


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
):
    """Configure logging for the evaluation pipeline."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def create_run_log_dir(prefix: str = "run") -> Path:
    """Create a timestamped directory for run logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "runs" / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================================
# Statistics Utilities
# ============================================================================


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance**0.5

    sorted_values = sorted(values)
    median = (
        sorted_values[n // 2]
        if n % 2 == 1
        else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "median": median,
    }


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    import math

    n = len(values)
    if n < 2:
        mean = values[0] if values else 0
        return (mean, mean)

    mean = sum(values) / n
    std = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5
    se = std / math.sqrt(n)

    # Z-score for 95% confidence
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    margin = z * se
    return (mean - margin, mean + margin)


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_celex(celex: str) -> bool:
    """Validate CELEX number format."""
    import re

    # Basic CELEX pattern
    pattern = r"^[0-9CE][0-9]{4}[A-Z]{1,2}[0-9]+$"
    return bool(re.match(pattern, celex))


def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> bool:
    """Validate query text."""
    if not query or not isinstance(query, str):
        return False
    query = query.strip()
    return min_length <= len(query) <= max_length


# ============================================================================
# Export Utilities
# ============================================================================


def export_to_csv(
    data: List[Dict],
    path: Path,
    columns: Optional[List[str]] = None,
):
    """Export list of dicts to CSV."""
    import csv

    if not data:
        return

    columns = columns or list(data[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def export_to_html_report(
    results: Dict,
    output_path: Path,
    title: str = "EUR-Lex Evaluation Report",
):
    """Generate HTML report from evaluation results."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .metric {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
"""

    # Add metrics section
    if "metrics" in results:
        html += "<h2>Retrieval Metrics</h2><table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in results["metrics"].items():
            html += f"<tr><td class='metric'>{metric}</td><td>{value:.4f}</td></tr>"
        html += "</table>"

    # Add success criteria
    if "success_criteria" in results:
        html += "<h2>Success Criteria</h2><table>"
        html += "<tr><th>Criterion</th><th>Status</th></tr>"
        for criterion, met in results["success_criteria"].items():
            status_class = "pass" if met else "fail"
            status_text = "PASS" if met else "FAIL"
            html += f"<tr><td>{criterion}</td><td class='{status_class}'>{status_text}</td></tr>"
        html += "</table>"

    html += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
