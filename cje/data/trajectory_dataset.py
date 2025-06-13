"""Dataset adapter for agent trajectories stored in JSONL.

Each line has at minimum:
    uid, step_idx, state, action, logp
Optional fields:
    reward  – per-step reward
    y_true  – terminal reward (can be repeated for last step)
    meta    – arbitrary dict

Lines are grouped by uid and ordered by step_idx to build `CJETrajectory`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional

from .base import CJEDataset
from .schema import CJEStep, CJETrajectory


class TrajectoryJSONLDataset(CJEDataset):
    """Load trajectories from a JSONL file.

    Parameters
    ----------
    file_path : str | Path
        Path to a JSONL file where each line is a dict representing an agent step.
    """

    name = "TrajectoryJSONL"

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)
        # Pre-load grouping index lazily
        self._cache: Optional[List[CJETrajectory]] = None

    # ------------------------------------------------------------------
    def _parse_file(self) -> List[CJETrajectory]:
        by_uid: Dict[str, List[Dict[str, Any]]] = {}
        with self.file_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = str(row.get("uid"))
                if uid not in by_uid:
                    by_uid[uid] = []
                by_uid[uid].append(row)

        trajectories: List[CJETrajectory] = []
        for uid, rows in by_uid.items():
            # sort by step_idx, default 0
            rows.sort(key=lambda r: r.get("step_idx", 0))
            steps: List[CJEStep] = []
            y_true: Optional[Any] = None
            for r in rows:
                step = CJEStep(
                    state=r.get("state"),
                    action=r.get("action"),
                    logp=float(r["logp"]),
                    reward=r.get("reward"),
                    meta=r.get("meta", {}),
                )
                steps.append(step)
                if r.get("y_true") is not None:
                    y_true = r["y_true"]
            trajectories.append(
                CJETrajectory(uid=uid, steps=steps, y_true=y_true, meta={})
            )
        return trajectories

    # ------------------------------------------------------------------
    def itersamples(self) -> Iterator[CJETrajectory]:  # type: ignore[override]
        if self._cache is None:
            self._cache = self._parse_file()
        for traj in self._cache:
            yield traj

    def __len__(self) -> int:  # type: ignore[override]
        if self._cache is None:
            self._cache = self._parse_file()
        return len(self._cache)

    # For compatibility with load_dataset() but not used now
    @classmethod
    def download(cls, cache_dir: Optional[str] = None, split: str = "train") -> "CJEDataset":  # type: ignore[override]
        raise NotImplementedError("Trajectory dataset needs explicit file path")
