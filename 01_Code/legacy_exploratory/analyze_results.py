"""
LEGACY / PROTOTYPE NOTE:
This script is an early version of the results analysis pipeline. 
The final evaluation logic and plotting are now handled by:
- compute_consistency.py
- analyze_consistency.py
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", default="results/summary_metrics.csv")
    parser.add_argument("--miou_csv", default="results/miou_metrics.csv")
    parser.add_argument("--out_dir", default="report_assets/quant/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Read files
    df_miou = pd.read_csv(args.miou_csv)
    df_summary = pd.read_csv(args.summary_csv)

    # 2. Parse severity
    def parse_severity(sev_str):
        if sev_str == "none":
            return 0
        elif isinstance(sev_str, str) and sev_str.startswith("level_"):
            return int(sev_str.replace("level_", ""))
        return sev_str

    df_miou['severity_int'] = df_miou['severity'].apply(parse_severity)
    df_summary['severity_int'] = df_summary['severity'].apply(parse_severity)

    # 3. Create summary table mean/std
    # mean±std across images per (condition, severity)
    grouped = df_miou.groupby(['condition', 'severity_int'])['mIoU']
    summary_stats = grouped.agg(['mean', 'std']).reset_index()
    # Format as mean±std 
    summary_stats['mean±std'] = summary_stats.apply(lambda row: f"{row['mean']:.4f}±{row['std']:.4f}" if pd.notnull(row['std']) else f"{row['mean']:.4f}±0.0000", axis=1)
    
    # Save directly
    summary_table_path = os.path.join(args.out_dir, "summary_table_mean_std.csv")
    summary_stats.to_csv(summary_table_path, index=False)

    # 4. Plots
    clean_val = df_summary[df_summary['condition'] == 'clean']['mIoU'].values[0]
    conditions = [c for c in df_summary['condition'].unique() if c != 'clean']

    # (A) Line plot: mIoU vs severity
    plt.figure(figsize=(8, 6))
    for cond in conditions:
        sub = df_summary[(df_summary['condition'] == cond) & (df_summary['severity_int'] > 0)].sort_values('severity_int')
        plt.plot(sub['severity_int'], sub['mIoU'], marker='o', label=cond)
    plt.axhline(clean_val, color='black', linestyle='--', label='clean')
    plt.xlabel("Severity")
    plt.ylabel("mIoU")
    plt.title("mIoU vs Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "miou_vs_severity.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, "miou_vs_severity.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # (A.1) Line plot: mIoU vs severity (with error bars)
    plt.figure(figsize=(8, 6))
    clean_stats = summary_stats[summary_stats['condition'] == 'clean']
    clean_mean = clean_stats['mean'].values[0]
    clean_std = clean_stats['std'].values[0] if pd.notnull(clean_stats['std'].values[0]) else 0.0

    for cond in conditions:
        sub = summary_stats[(summary_stats['condition'] == cond) & (summary_stats['severity_int'] > 0)].sort_values('severity_int')
        plt.errorbar(sub['severity_int'], sub['mean'], yerr=sub['std'], marker='o', label=cond, capsize=4)
        
    plt.axhline(clean_mean, color='black', linestyle='--', label='clean mean')
    plt.fill_between([1, 5], clean_mean - clean_std, clean_mean + clean_std, color='black', alpha=0.1, label='clean ± std')

    plt.xlabel("Severity")
    plt.ylabel("mIoU")
    plt.title("mIoU vs Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "miou_vs_severity_errorbars.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, "miou_vs_severity_errorbars.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # (B) Bar plot: mIoU at severity level 5 per corruption
    sev5 = df_summary[df_summary['severity_int'] == 5].copy()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sev5['condition'], sev5['mIoU'])
    plt.axhline(clean_val, color='black', linestyle='--', label='clean')
    plt.xlabel("Corruption")
    plt.ylabel("mIoU")
    plt.title("mIoU at Severity 5")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "miou_level5_bar.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, "miou_level5_bar.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # (C) Line plot: Δ from clean (clean−mIoU) vs severity
    plt.figure(figsize=(8, 6))
    for cond in conditions:
        sub = df_summary[(df_summary['condition'] == cond) & (df_summary['severity_int'] > 0)].sort_values('severity_int')
        delta = clean_val - sub['mIoU']
        plt.plot(sub['severity_int'], delta, marker='s', label=cond)
    plt.xlabel("Severity")
    plt.ylabel("Δ mIoU (clean - corrupted)")
    plt.title("mIoU Drop vs Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "delta_from_clean.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, "delta_from_clean.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Export Pivot Table
    df_corrupt = df_summary[df_summary['condition'] != 'clean']
    pivot = df_corrupt.pivot(index='condition', columns='severity', values='mIoU')
    col_order = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # Add clean baseline row
    pivot.loc['clean'] = [clean_val] + [pd.NA] * (len(pivot.columns) - 1)
    
    pivot_path = os.path.join(args.out_dir, "pivot_mean.csv")
    pivot.to_csv(pivot_path)

    # 6. Sanity Report
    print("=== Sanity Report ===")
    print(f"Clean baseline mIoU: {clean_val:.4f}")
    print(f"Conditions present : {', '.join(conditions)}")
    
    print("\nMonotonic decrease counts (out of 5 steps: clean->1->2->3->4->5):")
    for cond in conditions:
        sub = df_summary[(df_summary['condition'] == cond) & (df_summary['severity_int'] > 0)].sort_values('severity_int')
        vals = [clean_val] + sub['mIoU'].tolist()
        decrease_count = sum(vals[i] > vals[i+1] for i in range(len(vals)-1))
        print(f"  {cond.ljust(15)}: {decrease_count}/5")

    if not sev5.empty:
        worst_l5 = sev5.loc[sev5['mIoU'].idxmin()]
        print(f"\nWorst corruption at level 5: {worst_l5['condition']} (mIoU: {worst_l5['mIoU']:.4f})")

if __name__ == "__main__":
    main()
