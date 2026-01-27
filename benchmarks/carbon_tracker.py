"""
Unified Carbon Tracking Module for The Carbon Cost of Generative AI for Science.

This module provides a consistent interface for measuring energy consumption and
carbon emissions across all models in the benchmark. It wraps CodeCarbon with
fallback to manual timing when CodeCarbon is unavailable.

Usage:
    from benchmarks.carbon_tracker import CarbonTracker

    tracker = CarbonTracker(project_name="retrosynthesis_neuralsym")

    with tracker:
        # Your training or inference code here
        model.train()

    metrics = tracker.get_metrics()
    print(f"Energy: {metrics['energy_kwh']:.4f} kWh")
    print(f"CO2: {metrics['emissions_kg_co2']:.6f} kg")
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None


@dataclass
class HardwareInfo:
    """Hardware specification for reproducibility."""
    gpu_model: str = "Unknown"
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cpu_model: str = "Unknown"
    cpu_cores: int = 0
    ram_gb: float = 0.0
    cuda_version: str = "Unknown"
    platform: str = field(default_factory=lambda: platform.platform())

    @classmethod
    def auto_detect(cls) -> "HardwareInfo":
        """Automatically detect hardware configuration."""
        info = cls()
        info.platform = platform.platform()
        info.cpu_cores = os.cpu_count() or 0

        # Try to get CPU model
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                info.cpu_model = result.stdout.strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            info.cpu_model = line.split(":")[1].strip()
                            break
        except Exception:
            pass

        # Try to get GPU info via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,count,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(", ")
                    info.gpu_model = parts[0]
                    info.gpu_count = len(lines)
                    info.gpu_memory_gb = float(parts[2]) / 1024 if len(parts) > 2 else 0
        except Exception:
            pass

        # Try to get CUDA version
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.cuda_version = f"Driver {result.stdout.strip().split()[0]}"
        except Exception:
            pass

        # Try to get RAM
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                info.ram_gb = int(result.stdout.strip()) / (1024**3)
            elif platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            info.ram_gb = int(line.split()[1]) / (1024**2)
                            break
        except Exception:
            pass

        return info


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Energy and emissions
    energy_kwh: float = 0.0
    emissions_kg_co2: float = 0.0

    # Optional detailed metrics
    gpu_energy_kwh: float = 0.0
    cpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0

    # Task-specific metrics (filled by user)
    accuracy_top1: Optional[float] = None
    accuracy_top5: Optional[float] = None
    accuracy_top10: Optional[float] = None
    num_samples: Optional[int] = None
    batch_size: Optional[int] = None

    # Metadata
    project_name: str = ""
    model_name: str = ""
    task: str = ""  # "training" or "inference"
    hardware: Optional[Dict[str, Any]] = None


class CarbonTracker:
    """
    Unified carbon tracking wrapper.

    Supports CodeCarbon for detailed tracking with fallback to manual timing.

    Args:
        project_name: Identifier for the experiment
        output_dir: Directory to save results (default: benchmarks/results)
        model_name: Name of the model being benchmarked
        task: "training" or "inference"
        save_results: Whether to automatically save results to file
        country_iso_code: ISO code for carbon intensity (default: USA)

    Example:
        tracker = CarbonTracker(
            project_name="neuralsym_inference",
            model_name="neuralsym",
            task="inference"
        )

        with tracker:
            predictions = model.predict(test_data)

        tracker.add_accuracy(top1=0.455, top10=0.816, num_samples=5005)
        metrics = tracker.get_metrics()
        tracker.save()
    """

    def __init__(
        self,
        project_name: str,
        output_dir: str = "benchmarks/results",
        model_name: str = "",
        task: str = "inference",
        save_results: bool = True,
        country_iso_code: str = "USA"
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.task = task
        self.save_results = save_results
        self.country_iso_code = country_iso_code

        self._tracker: Optional[EmissionsTracker] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._metrics: Optional[BenchmarkMetrics] = None
        self._hardware: Optional[HardwareInfo] = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "CarbonTracker":
        """Start tracking."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop tracking."""
        self.stop()

    def start(self) -> None:
        """Start carbon tracking."""
        self._start_time = time.time()
        self._hardware = HardwareInfo.auto_detect()

        if CODECARBON_AVAILABLE:
            try:
                # CodeCarbon 3.x API - country_iso_code removed
                self._tracker = EmissionsTracker(
                    project_name=self.project_name,
                    output_dir=str(self.output_dir),
                    log_level="warning",
                    save_to_file=False,  # We save our own format
                )
                self._tracker.start()
            except Exception as e:
                print(f"Warning: CodeCarbon failed to start: {e}")
                print("Falling back to manual timing only.")
                self._tracker = None
        else:
            print("Note: CodeCarbon not installed. Using manual timing only.")
            print("Install with: pip install codecarbon")

    def stop(self) -> BenchmarkMetrics:
        """Stop tracking and collect metrics."""
        self._end_time = time.time()
        duration = self._end_time - self._start_time

        self._metrics = BenchmarkMetrics(
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(self._end_time).isoformat(),
            duration_seconds=duration,
            project_name=self.project_name,
            model_name=self.model_name,
            task=self.task,
            hardware=asdict(self._hardware) if self._hardware else None
        )

        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                if emissions is not None:
                    self._metrics.emissions_kg_co2 = emissions

                # Try to get detailed energy breakdown
                if hasattr(self._tracker, '_total_energy'):
                    energy = self._tracker._total_energy
                    self._metrics.energy_kwh = energy.kWh if hasattr(energy, 'kWh') else 0
                    if hasattr(energy, 'gpu_energy'):
                        self._metrics.gpu_energy_kwh = energy.gpu_energy
                    if hasattr(energy, 'cpu_energy'):
                        self._metrics.cpu_energy_kwh = energy.cpu_energy
                    if hasattr(energy, 'ram_energy'):
                        self._metrics.ram_energy_kwh = energy.ram_energy
            except Exception as e:
                print(f"Warning: Error collecting CodeCarbon metrics: {e}")

        if self.save_results:
            self.save()

        return self._metrics

    def add_accuracy(
        self,
        top1: Optional[float] = None,
        top5: Optional[float] = None,
        top10: Optional[float] = None,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """Add accuracy metrics to the results."""
        if self._metrics is None:
            raise RuntimeError("Tracker must be stopped before adding accuracy metrics")

        if top1 is not None:
            self._metrics.accuracy_top1 = top1
        if top5 is not None:
            self._metrics.accuracy_top5 = top5
        if top10 is not None:
            self._metrics.accuracy_top10 = top10
        if num_samples is not None:
            self._metrics.num_samples = num_samples
        if batch_size is not None:
            self._metrics.batch_size = batch_size

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        if self._metrics is None:
            raise RuntimeError("Tracker must be stopped before getting metrics")
        return asdict(self._metrics)

    def save(self, filename: Optional[str] = None) -> Path:
        """Save metrics to JSON file."""
        if self._metrics is None:
            raise RuntimeError("Tracker must be stopped before saving")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.project_name}_{timestamp}.json"

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(asdict(self._metrics), f, indent=2, default=str)

        print(f"Results saved to: {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print a summary of the benchmark results."""
        if self._metrics is None:
            raise RuntimeError("Tracker must be stopped before printing summary")

        print("\n" + "=" * 60)
        print(f"BENCHMARK SUMMARY: {self.project_name}")
        print("=" * 60)
        print(f"Model:    {self.model_name or 'N/A'}")
        print(f"Task:     {self.task}")
        print(f"Duration: {self._metrics.duration_seconds:.2f} seconds")
        print("-" * 60)
        print("ENERGY & EMISSIONS:")
        print(f"  Total Energy:  {self._metrics.energy_kwh:.6f} kWh")
        print(f"  CO2 Emissions: {self._metrics.emissions_kg_co2:.6f} kg")
        if self._metrics.gpu_energy_kwh > 0:
            print(f"  GPU Energy:    {self._metrics.gpu_energy_kwh:.6f} kWh")
        if self._metrics.cpu_energy_kwh > 0:
            print(f"  CPU Energy:    {self._metrics.cpu_energy_kwh:.6f} kWh")
        print("-" * 60)
        if self._metrics.accuracy_top1 is not None:
            print("ACCURACY:")
            print(f"  Top-1:  {self._metrics.accuracy_top1:.2%}")
            if self._metrics.accuracy_top5 is not None:
                print(f"  Top-5:  {self._metrics.accuracy_top5:.2%}")
            if self._metrics.accuracy_top10 is not None:
                print(f"  Top-10: {self._metrics.accuracy_top10:.2%}")
        if self._metrics.num_samples:
            print(f"  Samples: {self._metrics.num_samples}")
        print("=" * 60 + "\n")


def aggregate_results(results_dir: str = "benchmarks/results") -> List[Dict[str, Any]]:
    """
    Aggregate all benchmark results from JSON files.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of metrics dictionaries
    """
    results = []
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_source_file"] = str(json_file)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def create_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    Create a markdown comparison table from aggregated results.

    Args:
        results: List of metrics dictionaries from aggregate_results()

    Returns:
        Markdown-formatted table string
    """
    if not results:
        return "No results found."

    lines = [
        "| Model | Task | Top-1 | Top-10 | Energy (kWh) | CO2 (kg) | Duration (s) |",
        "|-------|------|-------|--------|--------------|----------|--------------|"
    ]

    for r in sorted(results, key=lambda x: x.get("model_name", "")):
        model = r.get("model_name", "N/A")
        task = r.get("task", "N/A")
        top1 = f"{r['accuracy_top1']:.1%}" if r.get("accuracy_top1") else "N/A"
        top10 = f"{r['accuracy_top10']:.1%}" if r.get("accuracy_top10") else "N/A"
        energy = f"{r['energy_kwh']:.4f}" if r.get("energy_kwh") else "N/A"
        co2 = f"{r['emissions_kg_co2']:.6f}" if r.get("emissions_kg_co2") else "N/A"
        duration = f"{r['duration_seconds']:.1f}" if r.get("duration_seconds") else "N/A"

        lines.append(f"| {model} | {task} | {top1} | {top10} | {energy} | {co2} | {duration} |")

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("Carbon Tracker Demo")
    print("-" * 40)

    tracker = CarbonTracker(
        project_name="demo_test",
        model_name="demo_model",
        task="inference",
        save_results=False
    )

    with tracker:
        # Simulate some work
        print("Running simulated workload...")
        time.sleep(2)

    tracker.add_accuracy(top1=0.455, top10=0.816, num_samples=5005)
    tracker.print_summary()
