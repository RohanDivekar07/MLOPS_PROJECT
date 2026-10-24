"""
Visualization Generator for Serving Benchmark Results.
Generates publication-quality charts for Throughput (RPS), Latency percentiles,
and Resource Utilization (CPU & RAM).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark", "results")

# Styling config
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold"
})

COLORS = {
    "TorchServe (REST / HTTP 1.1)": "#E63946",
    "TorchServe (gRPC / HTTP 2)": "#F4A261",
    "TF Serving (REST / HTTP 1.1)": "#457B9D",
    "TF Serving (gRPC / HTTP 2)": "#2A9D8F"
}

def plot_throughput_comparison(df, output_path):
    print("[*] Generating Throughput (RPS) comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    concurrencies = sorted(df["concurrency"].unique())
    x = np.arange(len(concurrencies))
    width = 0.18

    targets = df["target_name"].unique()
    for i, target in enumerate(targets):
        target_df = df[df["target_name"] == target].sort_values("concurrency")
        offset = (i - 1.5) * width
        color = COLORS.get(target, "#333333")
        bars = ax.bar(x + offset, target_df["rps"], width, label=target, color=color, edgecolor="black", linewidth=0.8, alpha=0.9)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Inference Throughput (Requests Per Second) vs Concurrency", pad=15)
    ax.set_xlabel("Concurrent Locust Users")
    ax.set_ylabel("Throughput (RPS) - Higher is Better")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} Users" for c in concurrencies])
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    ax.set_ylim(0, max(df["rps"]) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")

def plot_latency_percentiles(df, output_path):
    print("[*] Generating Latency Percentiles chart (Concurrency = 50)...")
    sub_df = df[df["concurrency"] == 50]
    if sub_df.empty:
        sub_df = df.iloc[:4]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    percentiles = ["latency_p50_ms", "latency_p90_ms", "latency_p95_ms", "latency_p99_ms"]
    perc_labels = ["P50 (Median)", "P90", "P95", "P99 (Tail)"]
    x = np.arange(len(percentiles))
    width = 0.18

    targets = sub_df["target_name"].unique()
    for i, target in enumerate(targets):
        tdf = sub_df[sub_df["target_name"] == target]
        if tdf.empty:
            continue
        row = tdf.iloc[0]
        vals = [row[p] for p in percentiles]
        offset = (i - 1.5) * width
        color = COLORS.get(target, "#333333")
        bars = ax.bar(x + offset, vals, width, label=target, color=color, edgecolor="black", linewidth=0.8, alpha=0.9)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}ms",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Response Latency Percentiles at 50 Concurrent Users (ms)", pad=15)
    ax.set_xlabel("Latency Metric")
    ax.set_ylabel("Response Latency (ms) - Lower is Better")
    ax.set_xticks(x)
    ax.set_xticklabels(perc_labels)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    ax.set_ylim(0, max(sub_df["latency_p99_ms"]) * 1.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")

def plot_resource_utilization(df, output_path):
    print("[*] Generating Resource Utilization comparison chart...")
    sub_df = df[df["concurrency"] == 50]
    if sub_df.empty:
        sub_df = df.iloc[:4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. CPU Utilization
    targets = sub_df["target_name"].values
    cpus = sub_df["cpu_avg_percent"].values
    colors = [COLORS.get(t, "#333333") for t in targets]

    bars1 = ax1.bar(np.arange(len(targets)), cpus, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax1.set_title("Average CPU Utilization (%)", pad=12)
    ax1.set_ylabel("CPU Usage (%) [Multi-core aggregate]")
    ax1.set_xticks(np.arange(len(targets)))
    ax1.set_xticklabels(["TorchServe\nREST", "TorchServe\ngRPC", "TF Serving\nREST", "TF Serving\ngRPC"], rotation=0)
    ax1.set_ylim(0, max(cpus) * 1.25)
    for b in bars1:
        ax1.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # 2. RAM Memory Footprint
    mems = sub_df["ram_avg_mb"].values
    bars2 = ax2.bar(np.arange(len(targets)), mems, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax2.set_title("Memory Footprint (RAM in MB)", pad=12)
    ax2.set_ylabel("Memory Allocated (MB)")
    ax2.set_xticks(np.arange(len(targets)))
    ax2.set_xticklabels(["TorchServe\nREST", "TorchServe\ngRPC", "TF Serving\nREST", "TF Serving\ngRPC"], rotation=0)
    ax2.set_ylim(0, max(mems) * 1.25)
    for b in bars2:
        ax2.annotate(f"{b.get_height():.0f} MB", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Resource Utilization Benchmark (50 Concurrent Clients)", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")

def plot_timeseries_metrics(results_dir, output_path):
    print("[*] Generating Timeseries CPU & Memory Usage chart...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

    target_map = {
        "torchserve_rest": ("TorchServe REST", COLORS["TorchServe (REST / HTTP 1.1)"]),
        "torchserve_grpc": ("TorchServe gRPC", COLORS["TorchServe (gRPC / HTTP 2)"]),
        "tfserving_rest": ("TF Serving REST", COLORS["TF Serving (REST / HTTP 1.1)"]),
        "tfserving_grpc": ("TF Serving gRPC", COLORS["TF Serving (gRPC / HTTP 2)"])
    }

    found = False
    for tid, (label, color) in target_map.items():
        csv_file = os.path.join(results_dir, f"metrics_{tid}.csv")
        if os.path.exists(csv_file):
            found = True
            mdf = pd.read_csv(csv_file)
            ax1.plot(mdf["elapsed_sec"], mdf["cpu_percent"], label=label, color=color, linewidth=2.0)
            ax2.plot(mdf["elapsed_sec"], mdf["memory_mb"], label=label, color=color, linewidth=2.0)

    if not found:
        print("[-] No timeseries metrics found to plot.")
        plt.close()
        return

    ax1.set_title("CPU Utilization Profile Over 60s Test Duration")
    ax1.set_ylabel("CPU (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("RAM Memory Footprint Profile Over 60s Test Duration")
    ax2.set_xlabel("Elapsed Time (seconds)")
    ax2.set_ylabel("RAM (MB)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved: {output_path}")

def main():
    summary_csv = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    if not os.path.exists(summary_csv):
        print("Summary CSV not found. Calling run_benchmark.py --generate-baseline...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "benchmark", "run_benchmark.py"), "--generate-baseline"])

    df = pd.read_csv(summary_csv)

    plot_throughput_comparison(df, os.path.join(RESULTS_DIR, "throughput_comparison.png"))
    plot_latency_percentiles(df, os.path.join(RESULTS_DIR, "latency_percentiles.png"))
    plot_resource_utilization(df, os.path.join(RESULTS_DIR, "resource_utilization.png"))
    plot_timeseries_metrics(RESULTS_DIR, os.path.join(RESULTS_DIR, "timeseries_resource_usage.png"))
    print("\n[+] All visual plots generated successfully in benchmark/results/")

if __name__ == "__main__":
    main()
