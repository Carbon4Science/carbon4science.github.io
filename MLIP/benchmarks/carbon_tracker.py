"""
Carbon Tracking Module for The Carbon Cost of Generative AI for Science.

This module provides a consistent interface for measuring energy consumption and
carbon emissions. It wraps CodeCarbon with fallback to manual timing.

Note: Task-specific metrics (accuracy, etc.) are handled by each task's evaluate.py.
This module focuses solely on carbon/energy measurement.

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
class CarbonMetrics:
    """Carbon and energy metrics collected during a benchmark run."""
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Energy and emissions (Wh and g for readable precision)
    energy_wh: float = 0.0
    emissions_g_co2: float = 0.0

    # Detailed energy breakdown (Wh)
    gpu_energy_wh: float = 0.0
    cpu_energy_wh: float = 0.0
    ram_energy_wh: float = 0.0

    # Peak resource usage
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0

    # Metadata
    project_name: str = ""
    model_name: str = ""
    task: str = ""  # "training" or "inference"
    hardware: Optional[Dict[str, Any]] = None


class CarbonTracker:
    """
    Carbon tracking wrapper for measuring energy consumption and CO2 emissions.

    Supports CodeCarbon for detailed tracking with fallback to manual timing.

    Args:
        project_name: Identifier for the experiment
        output_dir: Directory to save results (default: benchmarks/results)
        model_name: Name of the model being benchmarked
        task: "training" or "inference"
        save_results: Whether to automatically save results to file

    Example:
        tracker = CarbonTracker(
            project_name="neuralsym_inference",
            model_name="neuralsym",
            task="inference"
        )

        with tracker:
            predictions = model.predict(test_data)

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
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.task = task
        self.save_results = save_results

        self._tracker: Optional[EmissionsTracker] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._metrics: Optional[CarbonMetrics] = None
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

        # Reset GPU memory tracking
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        if CODECARBON_AVAILABLE:
            try:
                # CodeCarbon 3.x API
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

    @staticmethod
    def _get_peak_gpu_memory_mb() -> float:
        """Get peak GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def _get_peak_cpu_memory_mb() -> float:
        """Get peak CPU (RSS) memory usage in MB."""
        try:
            import resource
            # getrusage returns max RSS in KB on Linux
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            return rusage.ru_maxrss / 1024  # KB -> MB
        except Exception:
            pass
        return 0.0

    # Default carbon intensity: US grid average ~0.4 kgCO2/kWh = 0.4 g/Wh
    DEFAULT_CARBON_INTENSITY_G_PER_WH = 0.4

    # DRAM power estimate: ~0.375 W per GB (from DDR4/DDR5 specs)
    RAM_WATTS_PER_GB = 0.375

    @staticmethod
    def _estimate_gpu_energy_wh(duration_seconds: float) -> float:
        """Estimate GPU energy from nvidia-smi power draw (Wh)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Average power across all reporting GPUs (Watts)
                powers = [float(p.strip()) for p in lines if p.strip()]
                if powers:
                    avg_power_w = sum(powers) / len(powers)
                    return avg_power_w * duration_seconds / 3600  # W * s / 3600 = Wh
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _estimate_cpu_energy_wh(duration_seconds: float, cpu_cores: int = 0) -> float:
        """Estimate CPU energy from /proc/cpuinfo TDP and system load."""
        try:
            # Read 1-minute load average as fraction of total cores
            with open("/proc/loadavg") as f:
                load_1min = float(f.read().split()[0])
            if cpu_cores <= 0:
                cpu_cores = os.cpu_count() or 1
            utilization = min(load_1min / cpu_cores, 1.0)

            # Estimate TDP from cpuinfo (look for known keywords)
            # Fallback: assume 10W per core at full load (typical server CPU)
            tdp_w = cpu_cores * 10
            try:
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Thread(s) per core" in line:
                            threads_per_core = int(line.split(":")[1].strip())
                            physical_cores = cpu_cores // max(threads_per_core, 1)
                            tdp_w = physical_cores * 10  # ~10W per physical core
                            break
            except Exception:
                pass

            # Power = TDP × utilization (idle CPUs draw ~10-20% of TDP)
            power_w = tdp_w * max(utilization, 0.1)
            return power_w * duration_seconds / 3600
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _estimate_ram_energy_wh(duration_seconds: float, peak_ram_mb: float = 0.0) -> float:
        """Estimate RAM energy from peak memory usage."""
        if peak_ram_mb <= 0:
            return 0.0
        peak_ram_gb = peak_ram_mb / 1024
        power_w = peak_ram_gb * CarbonTracker.RAM_WATTS_PER_GB
        return power_w * duration_seconds / 3600

    def stop(self) -> CarbonMetrics:
        """Stop tracking and collect metrics."""
        self._end_time = time.time()
        duration = self._end_time - self._start_time

        # Collect peak memory usage
        peak_gpu_mb = self._get_peak_gpu_memory_mb()
        peak_cpu_mb = self._get_peak_cpu_memory_mb()

        self._metrics = CarbonMetrics(
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(self._end_time).isoformat(),
            duration_seconds=duration,
            peak_gpu_memory_mb=round(peak_gpu_mb, 1),
            peak_cpu_memory_mb=round(peak_cpu_mb, 1),
            project_name=self.project_name,
            model_name=self.model_name,
            task=self.task,
            hardware=asdict(self._hardware) if self._hardware else None
        )

        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                if emissions is not None:
                    # CodeCarbon returns kg; convert to g
                    self._metrics.emissions_g_co2 = emissions * 1000

                # Try to get detailed energy breakdown (CodeCarbon uses kWh; convert to Wh)
                if hasattr(self._tracker, '_total_energy'):
                    energy = self._tracker._total_energy
                    self._metrics.energy_wh = (energy.kWh * 1000) if hasattr(energy, 'kWh') else 0
                    if hasattr(energy, 'gpu_energy'):
                        self._metrics.gpu_energy_wh = energy.gpu_energy * 1000
                    if hasattr(energy, 'cpu_energy'):
                        self._metrics.cpu_energy_wh = energy.cpu_energy * 1000
                    if hasattr(energy, 'ram_energy'):
                        self._metrics.ram_energy_wh = energy.ram_energy * 1000
            except Exception as e:
                print(f"Warning: Error collecting CodeCarbon metrics: {e}")
        else:
            # Fallback: estimate energy from system sensors
            gpu_wh = self._estimate_gpu_energy_wh(duration)
            cpu_wh = self._estimate_cpu_energy_wh(
                duration, self._hardware.cpu_cores if self._hardware else 0
            )
            ram_wh = self._estimate_ram_energy_wh(duration, peak_cpu_mb)
            total_wh = gpu_wh + cpu_wh + ram_wh

            self._metrics.gpu_energy_wh = round(gpu_wh, 4)
            self._metrics.cpu_energy_wh = round(cpu_wh, 4)
            self._metrics.ram_energy_wh = round(ram_wh, 4)
            self._metrics.energy_wh = round(total_wh, 4)
            self._metrics.emissions_g_co2 = round(
                total_wh * self.DEFAULT_CARBON_INTENSITY_G_PER_WH, 4
            )

        if self.save_results:
            self.save()

        return self._metrics

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
        """Print a summary of the carbon metrics."""
        if self._metrics is None:
            raise RuntimeError("Tracker must be stopped before printing summary")

        print("\n" + "=" * 60)
        print(f"CARBON SUMMARY: {self.project_name}")
        print("=" * 60)
        print(f"Model:    {self.model_name or 'N/A'}")
        print(f"Task:     {self.task}")
        print(f"Duration: {self._metrics.duration_seconds:.2f} seconds")
        print("-" * 60)
        print("ENERGY & EMISSIONS:")
        print(f"  Total Energy:  {self._metrics.energy_wh:.4f} Wh")
        print(f"  CO2 Emissions: {self._metrics.emissions_g_co2:.4f} g")
        if self._metrics.gpu_energy_wh > 0:
            print(f"  GPU Energy:    {self._metrics.gpu_energy_wh:.4f} Wh")
        if self._metrics.cpu_energy_wh > 0:
            print(f"  CPU Energy:    {self._metrics.cpu_energy_wh:.4f} Wh")
        print("-" * 60)
        print("PEAK RESOURCE USAGE:")
        if self._metrics.peak_gpu_memory_mb > 0:
            print(f"  GPU Memory:    {self._metrics.peak_gpu_memory_mb:.1f} MB")
        if self._metrics.peak_cpu_memory_mb > 0:
            print(f"  CPU Memory:    {self._metrics.peak_cpu_memory_mb:.1f} MB")
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

    tracker.print_summary()
