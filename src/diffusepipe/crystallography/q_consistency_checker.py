"""Utility class that performs the q‑vector consistency test.

This was lost during the last refactor – the validator depends on it.  The
implementation below is **self‑contained**, uses only dxtbx/DIALS public APIs,
and does *zero* numpy/scitbx matrix gymnastics beyond basic linear algebra.

Algorithm (lab frame):
----------------------
1.  Pick up to *N* random reflections (default 500).
2.  For each reflection i:
    • `q_model` = s1_i – s0  (already in reflection table)
    • Obtain **observed** centroid `(x_px, y_px)` (prefer `xyzobs.px.value`; fall
      back to `xyzcal.px` if needed).
    • Convert that pixel to lab XYZ with `panel.get_pixel_lab_coord`.
    • Build `s1_obs` as the unit vector in that direction scaled by |s0|.
    • `q_obs` = s1_obs – s0.
    • Δq_i = |q_model – q_obs|.
3.  Aggregate mean / median / max and decide **pass** if:
      mean ≤ tolerance  **and**  max ≤ 5× tolerance.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["QConsistencyChecker"]


class QConsistencyChecker:
    """Stateless helper to keep the main validator lean."""

    # ------------------------------------------------------------------
    def check_q_consistency(
        self,
        experiment: object,
        reflections: object,
        tolerance: float = 0.01,
        max_reflections: int = 500,
    ) -> Tuple[bool, Dict[str, float]]:
        """Return (pass?, statistics) based on Δq magnitudes."""

        # Ensure required columns exist --------------------------------
        required_cols = {"miller_index", "panel", "s1"}
        if not required_cols.issubset(reflections):
            logger.warning("Reflection table missing required columns %s", required_cols)
            return False, {"count": 0, "mean": None, "median": None, "max": None}

        # Prefer observed pixel centroid columns ------------------------
        pos_cols_priority = ["xyzobs.px.value", "xyzcal.px"]
        pos_col = next((c for c in pos_cols_priority if c in reflections), None)
        if pos_col is None:
            logger.warning("No pixel‑centroid column found in reflection table")
            return False, {"count": 0, "mean": None, "median": None, "max": None}

        n_total = len(reflections)
        if n_total == 0:
            logger.warning("No reflections available for q‑consistency test")
            return False, {"count": 0, "mean": None, "median": None, "max": None}

        indices = random.sample(range(n_total), k=min(max_reflections, n_total))

        beam = experiment.beam
        detector = experiment.detector
        s0 = np.array(beam.get_s0())
        k_mod = np.linalg.norm(s0)

        delta_q = []
        for idx in indices:
            try:
                panel_id = int(reflections["panel"][idx])
                panel = detector[panel_id]

                # model q from reflection table (s1 column)
                q_model = np.array(reflections["s1"][idx]) - s0

                px, py, _ = reflections[pos_col][idx]
                lab_xyz = np.array(panel.get_pixel_lab_coord((px, py)))
                direction = lab_xyz / np.linalg.norm(lab_xyz)
                s1_obs = direction * k_mod
                q_obs = s1_obs - s0

                delta_q.append(np.linalg.norm(q_model - q_obs))
            except Exception as exc:
                logger.debug("Δq calc failed for refl %d: %s", idx, exc)
                continue

        if not delta_q:
            return False, {"count": 0, "mean": None, "median": None, "max": None}

        arr = np.array(delta_q)
        stats = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }
        passed = stats["mean"] <= tolerance and stats["max"] <= 5 * tolerance
        logger.info(
            "Q‑consistency: mean %.4g Å⁻¹, max %.4g Å⁻¹, n=%d, pass=%s",
            stats["mean"],
            stats["max"],
            stats["count"],
            passed,
        )
        return passed, stats