"""
comparison.py
-------------
Compare baseline vs improved model evaluation results.

Loads two JSON result files produced by evaluate.py and generates:
  - A structured comparison dict with delta metrics
  - A Markdown table report

Usage
-----
  python evaluation/comparison.py \\
      --baseline  outputs/evaluation/baseline_results.json \\
      --improved  outputs/evaluation/improved_results.json \\
      --output    outputs/evaluation/comparison_report.md
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def _safe_float(val) -> Optional[float]:
    """Return float or None for non-numeric values."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _delta_str(delta: Optional[float]) -> str:
    """Format a delta value with sign."""
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _wilcoxon_p_value(
    a_values: List[float], b_values: List[float]
) -> Optional[float]:
    """Compute Wilcoxon signed-rank p-value if scipy is available.

    Falls back to None (placeholder) if scipy is not installed or
    the sample size is too small.
    """
    try:
        from scipy.stats import wilcoxon
        if len(a_values) < 5 or len(b_values) < 5:
            return None
        common_len = min(len(a_values), len(b_values))
        stat, p = wilcoxon(a_values[:common_len], b_values[:common_len])
        return float(p)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001
        return None


# ===========================================================================
# ModelComparison
# ===========================================================================

class ModelComparison:
    """Compare evaluation results of a baseline model vs an improved model.

    Workflow
    --------
    1. ``load_results(baseline_path, improved_path)``
    2. ``compute_comparison_table()`` – returns a structured dict
    3. ``generate_comparison_report(output_path)`` – writes Markdown
    4. ``get_comparison_dict()`` – returns the structured dict directly
    """

    def __init__(self):
        self._baseline: Optional[dict] = None
        self._improved: Optional[dict] = None
        self._baseline_path: str = ""
        self._improved_path: str = ""
        self._comparison: Optional[dict] = None

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load_results(
        self,
        baseline_results_path: str,
        improved_results_path: str,
    ) -> None:
        """Load evaluation JSON files for both models.

        Parameters
        ----------
        baseline_results_path : str  Path to the baseline model JSON.
        improved_results_path : str  Path to the improved model JSON.

        Raises
        ------
        FileNotFoundError if either file does not exist.
        """
        for label, path_str in [
            ("baseline", baseline_results_path),
            ("improved", improved_results_path),
        ]:
            p = Path(path_str)
            if not p.exists():
                raise FileNotFoundError(
                    f"{label.capitalize()} results file not found: '{path_str}'"
                )

        with open(baseline_results_path, "r", encoding="utf-8") as f:
            self._baseline = json.load(f)
        with open(improved_results_path, "r", encoding="utf-8") as f:
            self._improved = json.load(f)

        self._baseline_path = baseline_results_path
        self._improved_path = improved_results_path
        self._comparison = None  # Invalidate cached comparison
        logger.info(
            "Loaded baseline (%d samples) and improved (%d samples) results.",
            self._baseline.get("n_samples", "?"),
            self._improved.get("n_samples", "?"),
        )

    def _require_loaded(self) -> None:
        if self._baseline is None or self._improved is None:
            raise RuntimeError(
                "Results not loaded. Call load_results() first."
            )

    # -----------------------------------------------------------------------
    # Comparison computation
    # -----------------------------------------------------------------------

    def compute_comparison_table(self) -> dict:
        """Compute delta metrics and p-values for all regions and metrics.

        Returns
        -------
        dict:
        {
          "regions": {
            "WT": {
              "metric_name": {
                "baseline": float,
                "improved": float,
                "delta": float,
                "p_value": float | None   # placeholder if scipy unavailable
              }, ...
            }, ...
          },
          "mean_dice": { "baseline", "improved", "delta", "p_value" },
          "baseline_path": str,
          "improved_path": str,
          "baseline_n_samples": int,
          "improved_n_samples": int,
        }
        """
        self._require_loaded()

        b_mean = self._baseline.get("mean", {})
        i_mean = self._improved.get("mean", {})

        # Per-patient lists for statistical tests
        b_patients = self._baseline.get("per_patient", [])
        i_patients = self._improved.get("per_patient", [])

        def _collect_per_patient(patients: list, key: str) -> List[float]:
            vals = []
            for p in patients:
                v = p.get(key, None)
                if v is None:
                    # Try nested: p['WT']['dice'] etc.
                    parts = key.split("/")
                    node = p
                    for part in parts:
                        if isinstance(node, dict):
                            node = node.get(part, None)
                        else:
                            node = None
                            break
                    v = node
                if v is not None:
                    f = _safe_float(v)
                    if f is not None:
                        vals.append(f)
            return vals

        regions = ["WT", "TC", "ET"]
        metric_names = ["dice", "iou", "precision", "recall", "specificity"]

        result: dict = {
            "baseline_path": self._baseline_path,
            "improved_path": self._improved_path,
            "baseline_n_samples": self._baseline.get("n_samples", 0),
            "improved_n_samples": self._improved.get("n_samples", 0),
            "regions": {},
        }

        for region in regions:
            result["regions"][region] = {}
            for metric in metric_names:
                # Mean values
                b_val = _safe_float(b_mean.get(f"{region}/{metric}"))
                i_val = _safe_float(i_mean.get(f"{region}/{metric}"))

                # Per-patient lists for Wilcoxon
                b_list = _collect_per_patient(b_patients, f"{region}/{metric}")
                i_list = _collect_per_patient(i_patients, f"{region}/{metric}")
                p_value = _wilcoxon_p_value(b_list, i_list)

                delta = (
                    (i_val - b_val) if (i_val is not None and b_val is not None) else None
                )

                result["regions"][region][metric] = {
                    "baseline": b_val,
                    "improved": i_val,
                    "delta": delta,
                    "p_value": p_value,
                }

        # Overall mean dice
        b_md = _safe_float(b_mean.get("mean_dice"))
        i_md = _safe_float(i_mean.get("mean_dice"))
        b_md_list = _collect_per_patient(b_patients, "mean_dice")
        i_md_list = _collect_per_patient(i_patients, "mean_dice")

        result["mean_dice"] = {
            "baseline": b_md,
            "improved": i_md,
            "delta": (i_md - b_md) if (i_md is not None and b_md is not None) else None,
            "p_value": _wilcoxon_p_value(b_md_list, i_md_list),
        }

        self._comparison = result
        return result

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def generate_comparison_report(self, output_path: str) -> None:
        """Write a Markdown comparison report to *output_path*.

        Parameters
        ----------
        output_path : str  Path for the output .md file.
        """
        if self._comparison is None:
            self.compute_comparison_table()

        cmp = self._comparison
        lines: List[str] = []

        lines += [
            "# Model Comparison Report: Baseline vs Improved",
            "",
            "## Overview",
            "",
            f"| | Baseline | Improved |",
            f"|---|---|---|",
            f"| Result file | `{Path(cmp['baseline_path']).name}` | `{Path(cmp['improved_path']).name}` |",
            f"| Samples | {cmp['baseline_n_samples']} | {cmp['improved_n_samples']} |",
            "",
            "## Per-Region Metrics",
            "",
        ]

        for region in ["WT", "TC", "ET"]:
            lines += [
                f"### {region} (Whole Tumour)" if region == "WT"
                else f"### {region} (Tumour Core)" if region == "TC"
                else f"### {region} (Enhancing Tumour)",
                "",
                "| Metric | Baseline | Improved | Δ (Improved − Baseline) | p-value |",
                "|--------|----------|----------|------------------------|---------|",
            ]
            for metric in ["dice", "iou", "precision", "recall", "specificity"]:
                entry = cmp["regions"][region][metric]
                b = f"{entry['baseline']:.4f}" if entry["baseline"] is not None else "N/A"
                i = f"{entry['improved']:.4f}" if entry["improved"] is not None else "N/A"
                d = _delta_str(entry["delta"])
                p = f"{entry['p_value']:.4f}" if entry["p_value"] is not None else "N/A*"
                lines.append(f"| {metric.capitalize()} | {b} | {i} | {d} | {p} |")
            lines.append("")

        # Summary row
        md_entry = cmp["mean_dice"]
        b_md = f"{md_entry['baseline']:.4f}" if md_entry["baseline"] is not None else "N/A"
        i_md = f"{md_entry['improved']:.4f}" if md_entry["improved"] is not None else "N/A"
        d_md = _delta_str(md_entry["delta"])
        p_md = f"{md_entry['p_value']:.4f}" if md_entry["p_value"] is not None else "N/A*"

        lines += [
            "## Overall Mean Dice",
            "",
            "| | Baseline | Improved | Δ | p-value |",
            "|---|---|---|---|---|",
            f"| Mean Dice (WT+TC+ET) | {b_md} | {i_md} | {d_md} | {p_md} |",
            "",
            "> \\* p-value requires `scipy` and ≥5 paired samples.",
            "",
        ]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Comparison report written → %s", out_path)

    # -----------------------------------------------------------------------
    # API helper
    # -----------------------------------------------------------------------

    def get_comparison_dict(self) -> dict:
        """Return the full structured comparison dict.

        Computes it first if not yet computed.

        Returns
        -------
        dict produced by :meth:`compute_comparison_table`.
        """
        if self._comparison is None:
            self.compute_comparison_table()
        return self._comparison


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compare baseline vs improved model evaluation results."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline evaluation_results.json",
    )
    parser.add_argument(
        "--improved",
        type=str,
        required=True,
        help="Path to improved evaluation_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/comparison_report.md",
        help="Output path for the Markdown comparison report.",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Optional path to also save the comparison dict as JSON.",
    )
    args = parser.parse_args()

    cmp = ModelComparison()
    cmp.load_results(args.baseline, args.improved)
    cmp.compute_comparison_table()
    cmp.generate_comparison_report(args.output)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cmp.get_comparison_dict(), f, indent=2)
        logger.info("JSON comparison saved → %s", out)

    # Print summary to stdout
    d = cmp.get_comparison_dict()
    md = d.get("mean_dice", {})
    print(
        f"\nMean Dice: Baseline={md.get('baseline', 'N/A')}  "
        f"Improved={md.get('improved', 'N/A')}  "
        f"Δ={_delta_str(md.get('delta'))}"
    )


if __name__ == "__main__":
    main()
