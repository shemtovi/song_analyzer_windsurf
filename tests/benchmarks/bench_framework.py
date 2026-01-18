"""Benchmark framework for measuring and tracking performance.

This module provides tools for:
- Timing individual pipeline stages
- Measuring memory usage
- Detecting performance regressions
- Generating benchmark reports
"""

import time
import json
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import functools


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""

    stage: str
    duration_seconds: float
    input_duration_seconds: float = 0.0
    notes_detected: int = 0
    peak_memory_mb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def realtime_ratio(self) -> float:
        """How many times faster than realtime (>1 is good)."""
        if self.duration_seconds <= 0 or self.input_duration_seconds <= 0:
            return 0.0
        return self.input_duration_seconds / self.duration_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["realtime_ratio"] = self.realtime_ratio
        return result


@dataclass
class BenchmarkReport:
    """Complete benchmark report with all stages."""

    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    stages: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, str] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Total time across all stages."""
        return sum(r.duration_seconds for r in self.stages)

    @property
    def total_notes(self) -> int:
        """Total notes detected across all stages."""
        return sum(r.notes_detected for r in self.stages)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "stages": [r.to_dict() for r in self.stages],
            "system_info": self.system_info,
            "total_time": self.total_time,
            "total_notes": self.total_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.write_text(self.to_json())


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def get_system_info() -> Dict[str, str]:
    """Get system information for benchmark context."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = "N/A"

    return info


class BenchmarkRunner:
    """Runner for timing pipeline stages and generating reports."""

    def __init__(
        self,
        name: str = "benchmark",
        baseline_path: Optional[Path] = None,
    ):
        """Initialize benchmark runner.

        Args:
            name: Name for this benchmark run
            baseline_path: Path to baseline JSON for regression detection
        """
        self.name = name
        self.baseline_path = baseline_path or Path("tests/benchmarks/baselines.json")
        self.results: List[BenchmarkResult] = []
        self._start_memory = get_memory_usage_mb()

    def time_stage(
        self,
        stage_name: str,
        func: Callable,
        *args,
        input_duration: float = 0.0,
        **kwargs,
    ) -> Any:
        """Time a stage and record the result.

        Args:
            stage_name: Name of the stage being timed
            func: Function to execute
            *args: Arguments to pass to func
            input_duration: Duration of input audio in seconds
            **kwargs: Keyword arguments to pass to func

        Returns:
            The result of calling func(*args, **kwargs)
        """
        start_memory = get_memory_usage_mb()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        duration = time.perf_counter() - start_time
        end_memory = get_memory_usage_mb()

        # Count notes if result is a list of notes
        notes_detected = 0
        if isinstance(result, list):
            notes_detected = len(result)
        elif hasattr(result, "__len__"):
            notes_detected = len(result)

        benchmark_result = BenchmarkResult(
            stage=stage_name,
            duration_seconds=duration,
            input_duration_seconds=input_duration,
            notes_detected=notes_detected,
            peak_memory_mb=max(0, end_memory - start_memory),
        )

        self.results.append(benchmark_result)
        return result

    def generate_report(self) -> BenchmarkReport:
        """Generate a benchmark report from recorded results."""
        return BenchmarkReport(
            name=self.name,
            stages=self.results.copy(),
            system_info=get_system_info(),
        )

    def save_report(self, path: Path) -> None:
        """Generate and save report to file."""
        report = self.generate_report()
        report.save(path)

    def check_regression(
        self,
        threshold_percent: float = 20.0,
    ) -> List[str]:
        """Check if any stage regressed beyond threshold.

        Args:
            threshold_percent: Maximum allowed slowdown percentage

        Returns:
            List of regression messages (empty if no regressions)
        """
        if not self.baseline_path.exists():
            return []

        try:
            baselines = json.loads(self.baseline_path.read_text())
        except (json.JSONDecodeError, IOError):
            return []

        regressions = []
        baseline_stages = {s["stage"]: s for s in baselines.get("stages", [])}

        for result in self.results:
            if result.stage in baseline_stages:
                baseline = baseline_stages[result.stage]["duration_seconds"]
                if baseline <= 0:
                    continue

                slowdown = (result.duration_seconds / baseline - 1) * 100

                if slowdown > threshold_percent:
                    regressions.append(
                        f"{result.stage}: {result.duration_seconds:.2f}s vs "
                        f"baseline {baseline:.2f}s (+{slowdown:.1f}%)"
                    )

        return regressions

    def print_summary(self) -> None:
        """Print a summary of benchmark results to console."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")

        for result in self.results:
            ratio_str = ""
            if result.realtime_ratio > 0:
                ratio_str = f" ({result.realtime_ratio:.1f}x realtime)"

            notes_str = ""
            if result.notes_detected > 0:
                notes_str = f", {result.notes_detected} notes"

            mem_str = ""
            if result.peak_memory_mb > 0:
                mem_str = f", {result.peak_memory_mb:.1f}MB"

            print(f"  {result.stage}: {result.duration_seconds:.3f}s{ratio_str}{notes_str}{mem_str}")

        print(f"{'='*60}")
        print(f"Total: {sum(r.duration_seconds for r in self.results):.3f}s")
        print()


def benchmark(name: str = None):
    """Decorator to benchmark a function.

    Usage:
        @benchmark("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stage_name = name or func.__name__
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            print(f"[BENCHMARK] {stage_name}: {duration:.3f}s")
            return result
        return wrapper
    return decorator
