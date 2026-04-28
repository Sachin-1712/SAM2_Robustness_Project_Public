"""
analyze_consistency.py — Analysis & Plotting for Clean-vs-Corrupted Consistency

Reads the consistency metric CSVs produced by compute_consistency.py and
generates dissertation-ready plots, tables, and a sanity report.

This is the PRIMARY analysis script. For the secondary GT-based analysis,
see analyze_results.py.

Outputs:
    Tables:
        - summary_table.csv          — mean ± std per (condition, severity)

    Plots (saved as both PNG and PDF):
        - foreground_iou_vs_severity          — line plot
        - foreground_iou_vs_severity_errorbars — error-bar line plot
        - consistency_drop_vs_severity        — drop = 1.0 − IoU
        - foreground_iou_level_max_bar        — bar chart at highest severity
        - best_match_iou_vs_severity          — optional: instance-level metric
        - mask_count_diff_vs_severity         — optional: mask count divergence

Usage:
    python scripts/analyze_consistency.py \\
        --metrics_csv runs/run3_diffusion_eval/results/consistency_metrics_5img.csv \\
        --summary_csv runs/run3_diffusion_eval/results/summary_consistency_5img.csv \\
        --out_dir runs/run3_diffusion_eval/report_assets/quant_consistency_5img

Author: Sachin (Final Year Project — SAM2 Robustness under Adverse Weather)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Plotting Style
# ──────────────────────────────────────────────────────────────────────

# Professional, dissertation-friendly defaults
plt.rcParams.update({
    "figure.figsize": (8, 5.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

# Colour palette — distinct, colour-blind-friendly, professional
CONDITION_COLORS = {
    "rain": "#2196F3",       # blue
    "fog": "#9E9E9E",        # grey
    "frost": "#00BCD4",      # cyan
    "snow": "#7E57C2",       # purple
    "brightness": "#FF9800", # amber
}

CONDITION_MARKERS = {
    "rain": "o",
    "fog": "s",
    "frost": "D",
    "snow": "^",
    "brightness": "v",
}


def _get_color(condition):
    return CONDITION_COLORS.get(condition, "#333333")


def _get_marker(condition):
    return CONDITION_MARKERS.get(condition, "o")


def _save_fig(fig, out_dir, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    try:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    except Exception as e:
        print(f"    [Warning] Could not save PDF for {name} due to system policy/missing DLL: {e}")
    plt.close(fig)
    print(f"    Saved: {name}.png / .pdf")


# ──────────────────────────────────────────────────────────────────────
# Summary Table
# ──────────────────────────────────────────────────────────────────────

def make_summary_table(df, out_dir):
    """
    Create a formatted summary CSV with mean ± std per (condition, severity).
    """
    grouped = df.groupby(["condition", "severity"])
    metrics = [
        "foreground_iou", "foreground_dice",
        "clean_foreground_preserved_ratio",
        "best_match_mask_iou_mean", "mask_count_difference",
    ]

    rows = []
    for (cond, sev), grp in sorted(grouped):
        row = {"condition": cond, "severity": sev, "n_images": len(grp)}
        for m in metrics:
            vals = grp[m].dropna()
            if len(vals) > 0:
                mean_v = vals.mean()
                std_v = vals.std(ddof=0)  # population std for small N
                row[f"{m}_mean"] = round(mean_v, 6)
                row[f"{m}_std"] = round(std_v, 6)
                row[f"{m}_formatted"] = f"{mean_v:.4f} ± {std_v:.4f}"
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
                row[f"{m}_formatted"] = "N/A"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    table_path = os.path.join(out_dir, "summary_table.csv")
    summary_df.to_csv(table_path, index=False)
    print(f"    Saved: summary_table.csv")
    return summary_df


# ──────────────────────────────────────────────────────────────────────
# Plot: Foreground IoU vs Severity (line)
# ──────────────────────────────────────────────────────────────────────

def plot_iou_vs_severity(df, out_dir):
    """Line plot of mean foreground IoU per condition across severity levels."""
    fig, ax = plt.subplots()

    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())

    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        means = sub.groupby("severity")["foreground_iou"].mean().reindex(severities)
        ax.plot(
            severities, means.values,
            marker=_get_marker(cond), color=_get_color(cond),
            label=cond.capitalize(),
        )

    # Clean baseline reference line at IoU = 1.0
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfect consistency")

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Foreground Consistency IoU")
    ax.set_title("Foreground Consistency IoU vs Corruption Severity")
    ax.set_xticks(severities)
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(loc="lower left")

    _save_fig(fig, out_dir, "foreground_iou_vs_severity")


# ──────────────────────────────────────────────────────────────────────
# Plot: Foreground IoU vs Severity (error bars)
# ──────────────────────────────────────────────────────────────────────

def plot_iou_vs_severity_errorbars(df, out_dir):
    """Error-bar line plot showing mean ± std foreground IoU."""
    fig, ax = plt.subplots()

    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())

    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        grouped = sub.groupby("severity")["foreground_iou"]
        means = grouped.mean().reindex(severities)
        stds = grouped.std(ddof=0).reindex(severities).fillna(0)

        ax.errorbar(
            severities, means.values, yerr=stds.values,
            marker=_get_marker(cond), color=_get_color(cond),
            label=cond.capitalize(), capsize=4,
        )

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfect consistency")

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Foreground Consistency IoU")
    ax.set_title("Foreground Consistency IoU vs Corruption Severity (Mean ± Std)")
    ax.set_xticks(severities)
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(loc="lower left")

    _save_fig(fig, out_dir, "foreground_iou_vs_severity_errorbars")


# ──────────────────────────────────────────────────────────────────────
# Plot: Consistency Drop vs Severity
# ──────────────────────────────────────────────────────────────────────

def plot_consistency_drop(df, out_dir):
    """Line plot of consistency drop (1.0 − IoU) vs severity."""
    fig, ax = plt.subplots()

    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())

    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        means = sub.groupby("severity")["foreground_iou"].mean().reindex(severities)
        drops = 1.0 - means.values

        ax.plot(
            severities, drops,
            marker=_get_marker(cond), color=_get_color(cond),
            label=cond.capitalize(),
        )

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Consistency Drop (1 − IoU)")
    ax.set_title("Robustness Drop Relative to Clean")
    ax.set_xticks(severities)
    ax.set_ylim(bottom=-0.02)
    ax.legend(loc="upper left")

    _save_fig(fig, out_dir, "consistency_drop_vs_severity")


# ──────────────────────────────────────────────────────────────────────
# Plot: Bar Chart at Highest Severity
# ──────────────────────────────────────────────────────────────────────

def plot_iou_bar_max_severity(df, out_dir):
    """Bar chart of foreground IoU at the highest severity level."""
    max_sev = df["severity"].max()

    sub = df[df["severity"] == max_sev].copy()
    means = sub.groupby("condition")["foreground_iou"].mean().sort_index()

    fig, ax = plt.subplots()
    bar_colors = [_get_color(c) for c in means.index]
    bars = ax.bar(
        [c.capitalize() for c in means.index], means.values,
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfect consistency")
    ax.set_ylabel("Foreground Consistency IoU")
    ax.set_title(f"Foreground Consistency IoU at Severity {max_sev}")
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    _save_fig(fig, out_dir, f"foreground_iou_level{max_sev}_bar")


# ──────────────────────────────────────────────────────────────────────
# Plot: Best-Match Mask IoU vs Severity (optional)
# ──────────────────────────────────────────────────────────────────────

def plot_best_match_iou(df, out_dir):
    """Line plot of mean best-match mask IoU vs severity."""
    if "best_match_mask_iou_mean" not in df.columns:
        return

    fig, ax = plt.subplots()

    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())

    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        means = sub.groupby("severity")["best_match_mask_iou_mean"].mean().reindex(severities)
        ax.plot(
            severities, means.values,
            marker=_get_marker(cond), color=_get_color(cond),
            label=cond.capitalize(),
        )

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Best-Match Mask IoU (mean)")
    ax.set_title("Instance-Level Best-Match IoU vs Corruption Severity")
    ax.set_xticks(severities)
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(loc="lower left")

    _save_fig(fig, out_dir, "best_match_iou_vs_severity")


# ──────────────────────────────────────────────────────────────────────
# Plot: Mask Count Difference vs Severity (optional)
# ──────────────────────────────────────────────────────────────────────

def plot_mask_count_diff(df, out_dir):
    """Line plot of mean mask count difference vs severity."""
    if "mask_count_difference" not in df.columns:
        return

    fig, ax = plt.subplots()

    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())

    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        means = sub.groupby("severity")["mask_count_difference"].mean().reindex(severities)
        ax.plot(
            severities, means.values,
            marker=_get_marker(cond), color=_get_color(cond),
            label=cond.capitalize(),
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.5, label="No change")

    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Mask Count Difference (corrupt − clean)")
    ax.set_title("Mask Count Change vs Corruption Severity")
    ax.set_xticks(severities)
    ax.legend(loc="best")

    _save_fig(fig, out_dir, "mask_count_diff_vs_severity")


# ──────────────────────────────────────────────────────────────────────
# Sanity Report
# ──────────────────────────────────────────────────────────────────────

def print_sanity_report(df):
    """Print a concise sanity report to the console."""
    conditions = sorted(df["condition"].unique())
    severities = sorted(df["severity"].unique())
    max_sev = max(severities)
    n_pairs = len(df)

    print()
    print("=" * 60)
    print("  SANITY REPORT")
    print("=" * 60)
    print(f"  Conditions present:   {', '.join(conditions)}")
    print(f"  Severity levels:      {severities}")
    print(f"  Total image pairs:    {n_pairs}")

    # Worst condition at highest severity
    max_sev_data = df[df["severity"] == max_sev]
    if not max_sev_data.empty:
        worst = max_sev_data.groupby("condition")["foreground_iou"].mean()
        worst_cond = worst.idxmin()
        worst_val = worst.min()
        print(f"  Worst at L{max_sev}:        {worst_cond} "
              f"(mean IoU = {worst_val:.4f})")

    # General trend check: does IoU decrease with severity?
    print()
    print("  Monotonic decrease check (IoU should drop with severity):")
    for cond in conditions:
        sub = df[df["condition"] == cond]
        means = sub.groupby("severity")["foreground_iou"].mean().sort_index()
        values = means.values
        is_decreasing = all(
            values[i] >= values[i + 1] for i in range(len(values) - 1)
        )
        trend = "✓ decreasing" if is_decreasing else "✗ non-monotonic"
        vals_str = " → ".join(f"{v:.4f}" for v in values)
        print(f"    {cond.ljust(12)}: {vals_str}  [{trend}]")

    print("=" * 60)
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate dissertation-ready plots and tables from SAM2 "
            "foreground consistency metrics. This is the PRIMARY analysis "
            "script for the robustness study."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/analyze_consistency.py \\
      --metrics_csv runs/run3_diffusion_eval/results/consistency_metrics_5img.csv \\
      --out_dir runs/run3_diffusion_eval/report_assets/quant_consistency_5img

  # With explicit summary CSV
  python scripts/analyze_consistency.py \\
      --metrics_csv results/consistency_metrics_5img.csv \\
      --summary_csv results/summary_consistency_5img.csv \\
      --out_dir report_assets/quant_consistency_5img
""",
    )

    parser.add_argument(
        "--metrics_csv", type=str, required=True,
        help="Path to the per-image consistency metrics CSV "
             "(from compute_consistency.py).",
    )
    parser.add_argument(
        "--summary_csv", type=str, default=None,
        help="Optional path to the summary CSV (not required for plotting, "
             "but will be regenerated and saved to --out_dir).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="report_assets/quant_consistency_5img",
        help="Output directory for plots and tables.",
    )

    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    if not os.path.exists(args.metrics_csv):
        print(f"ERROR: Metrics CSV not found: {args.metrics_csv}")
        sys.exit(1)

    df = pd.read_csv(args.metrics_csv)

    if df.empty:
        print("ERROR: Metrics CSV is empty.")
        sys.exit(1)

    # Ensure severity is integer
    df["severity"] = df["severity"].astype(int)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Startup ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SAM2 Consistency Analysis & Plotting")
    print("=" * 60)
    print(f"  Input CSV:       {args.metrics_csv}")
    print(f"  Output dir:      {args.out_dir}")
    print(f"  Rows loaded:     {len(df)}")
    print(f"  Conditions:      {sorted(df['condition'].unique())}")
    print(f"  Severities:      {sorted(df['severity'].unique())}")
    print("=" * 60)
    print()

    # ── Generate outputs ──────────────────────────────────────────
    print("  Generating outputs:")

    # Summary table
    summary_df = make_summary_table(df, args.out_dir)

    # Main plots
    plot_iou_vs_severity(df, args.out_dir)
    plot_iou_vs_severity_errorbars(df, args.out_dir)
    plot_consistency_drop(df, args.out_dir)
    plot_iou_bar_max_severity(df, args.out_dir)

    # Optional plots
    plot_best_match_iou(df, args.out_dir)
    plot_mask_count_diff(df, args.out_dir)

    # ── Sanity report ─────────────────────────────────────────────
    print_sanity_report(df)

    print("✅ Consistency analysis complete!")


if __name__ == "__main__":
    main()
