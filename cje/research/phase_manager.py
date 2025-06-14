"""
Research Phase Manager

Coordinates research-specific phases that extend beyond the base CJE pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class PhaseResult:
    """Result from a single research phase."""

    phase_name: str
    success: bool
    data: Dict[str, Any]
    output_files: List[str]
    metadata: Dict[str, Any]


class ResearchPhaseManager:
    """
    Manages additional research phases beyond base CJE pipeline.

    Coordinates data flow between phases and handles phase-specific
    outputs and caching.
    """

    def __init__(self, work_dir: Path, base_results: Dict[str, Any]):
        self.work_dir = work_dir
        self.base_results = base_results
        self.phase_results: Dict[str, Any] = {}

        # Create phase-specific directories
        self.phase_dirs = {
            "oracle_analysis": work_dir / "oracle_analysis",
            "gold_validation": work_dir / "gold_validation",
            "diagnostics": work_dir / "diagnostics",
            "deliverables": work_dir / "deliverables",
        }

        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)

    def get_phase_dir(self, phase_name: str) -> Path:
        """Get directory for a specific phase."""
        return self.phase_dirs.get(phase_name, self.work_dir / phase_name)

    def save_phase_result(self, phase_name: str, result: PhaseResult) -> None:
        """Save result from a research phase."""
        self.phase_results[phase_name] = result

        # Save to disk
        result_file = self.get_phase_dir(phase_name) / f"{phase_name}_result.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "phase_name": result.phase_name,
                    "success": result.success,
                    "data": result.data,
                    "output_files": result.output_files,
                    "metadata": result.metadata,
                },
                f,
                indent=2,
            )

    def load_phase_result(self, phase_name: str) -> PhaseResult:
        """Load cached result from a research phase."""
        result_file = self.get_phase_dir(phase_name) / f"{phase_name}_result.json"

        if not result_file.exists():
            raise FileNotFoundError(f"No cached result for phase: {phase_name}")

        with open(result_file) as f:
            data = json.load(f)

        return PhaseResult(
            phase_name=data["phase_name"],
            success=data["success"],
            data=data["data"],
            output_files=data["output_files"],
            metadata=data["metadata"],
        )

    def has_phase_result(self, phase_name: str) -> bool:
        """Check if phase result exists and is cached."""
        result_file = self.get_phase_dir(phase_name) / f"{phase_name}_result.json"
        return result_file.exists()

    def get_oracle_data(self) -> Dict[str, Any]:
        """Extract oracle data from base CJE results."""
        # CJE stores oracle data in the work directory
        oracle_files = list(self.work_dir.glob("*oracle*"))

        if not oracle_files:
            console.print(
                "[yellow]⚠️  No oracle files found, checking base results[/yellow]"
            )
            return {"oracle_data": [], "source": "base_results"}

        # Load the most recent oracle file
        latest_oracle = max(oracle_files, key=lambda p: p.stat().st_mtime)
        console.print(f"[blue]Loading oracle data from: {latest_oracle.name}[/blue]")

        try:
            with open(latest_oracle) as f:
                if latest_oracle.suffix == ".jsonl":
                    oracle_data = [json.loads(line) for line in f]
                else:
                    oracle_data = json.load(f)

            return {
                "oracle_data": oracle_data,
                "source": str(latest_oracle),
                "sample_count": (
                    len(oracle_data) if isinstance(oracle_data, list) else 1
                ),
            }
        except Exception as e:
            console.print(f"[red]Failed to load oracle data: {e}[/red]")
            return {"oracle_data": [], "source": "error", "error": str(e)}

    def get_policy_results(self) -> Dict[str, Any]:
        """Extract policy evaluation results from base CJE."""
        # Base results should contain policy estimates
        policy_results = {}

        for key, value in self.base_results.items():
            if isinstance(value, dict) and "estimate" in value:
                policy_results[key] = value

        return policy_results

    def generate_phase_summary(self) -> Dict[str, Any]:
        """Generate summary of all research phases."""
        summary = {
            "base_results_summary": {
                "policies_evaluated": len(self.get_policy_results()),
                "oracle_enabled": "oracle" in str(self.work_dir),
                "work_directory": str(self.work_dir),
            },
            "research_phases": {},
        }

        for phase_name, result in self.phase_results.items():
            summary["research_phases"][phase_name] = {
                "success": result.success,
                "output_files": len(result.output_files),
                "metadata": result.metadata,
            }

        return summary

    def export_all_data(self, output_file: Path) -> None:
        """Export all phase data to a single comprehensive file."""
        export_data = {
            "base_results": self.base_results,
            "oracle_data": self.get_oracle_data(),
            "policy_results": self.get_policy_results(),
            "phase_results": {
                name: {
                    "success": result.success,
                    "data": result.data,
                    "metadata": result.metadata,
                }
                for name, result in self.phase_results.items()
            },
            "phase_summary": self.generate_phase_summary(),
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        console.print(f"[green]📦 All phase data exported to: {output_file}[/green]")
