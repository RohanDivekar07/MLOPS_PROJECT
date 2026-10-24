"""
Resource Utilization Monitor for Enterprise Model Serving Benchmark.
Samples and records CPU and RAM consumption during benchmark runs into CSV files.
"""

import os
import sys
import time
import csv
import threading
import subprocess
import psutil

class ResourceMonitor:
    def __init__(self, output_file, interval=1.0, container_name=None):
        self.output_file = output_file
        self.interval = interval
        self.container_name = container_name
        self._stop_event = threading.Event()
        self._thread = None
        self.data_points = []

    def _collect_docker_stats(self):
        """Attempts to poll docker container stats directly if available."""
        if not self.container_name:
            return None
        try:
            cmd = [
                "docker", "stats", "--no-stream",
                "--format", "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}",
                self.container_name
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().split(",")
                cpu_str = parts[0].replace("%", "").strip()
                mem_usage_str = parts[1].split("/")[0].strip()
                mem_perc_str = parts[2].replace("%", "").strip()
                
                # Parse memory MB
                mem_mb = 0.0
                if "GiB" in mem_usage_str:
                    mem_mb = float(mem_usage_str.replace("GiB", "").strip()) * 1024
                elif "MiB" in mem_usage_str:
                    mem_mb = float(mem_usage_str.replace("MiB", "").strip())
                elif "kB" in mem_usage_str:
                    mem_mb = float(mem_usage_str.replace("kB", "").strip()) / 1024

                return float(cpu_str), mem_mb, float(mem_perc_str)
        except Exception:
            return None
        return None

    def _monitor_loop(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "elapsed_sec", "cpu_percent", "memory_mb", "memory_percent"])

            start_t = time.time()
            # Initialize psutil cpu percent
            psutil.cpu_percent(interval=None)

            while not self._stop_event.is_set():
                now = time.time()
                elapsed = round(now - start_t, 2)
                
                docker_stat = self._collect_docker_stats()
                if docker_stat:
                    cpu_perc, mem_mb, mem_perc = docker_stat
                else:
                    cpu_perc = psutil.cpu_percent(interval=None)
                    mem_info = psutil.virtual_memory()
                    mem_mb = round((mem_info.total - mem_info.available) / (1024 * 1024), 2)
                    mem_perc = mem_info.percent

                row = [round(now, 2), elapsed, cpu_perc, mem_mb, mem_perc]
                self.data_points.append(row)
                writer.writerow(row)
                f.flush()

                self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[*] Started Resource Monitor -> {self.output_file}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        print(f"[*] Stopped Resource Monitor. Captured {len(self.data_points)} data points.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resource Monitor CLI")
    parser.add_argument("--output", type=str, default="benchmark/results/resource_sample.csv")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--container", type=str, default=None, help="Container name to monitor")
    args = parser.parse_args()

    mon = ResourceMonitor(args.output, interval=1.0, container_name=args.container)
    mon.start()
    try:
        time.sleep(args.duration)
    finally:
        mon.stop()
