"""
Live tracking and reporting for MemoryLLM bulk tests.

Features:
- Real-time logging to files
- Winner detection and alerts
- Live leaderboard updates
- Optional webhook notifications (Discord/Slack/Telegram)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable
import threading


@dataclass
class LiveResult:
    """Single test result with metadata."""
    strategy: str
    accuracy: float
    time_seconds: float
    num_contexts: int
    num_samples: int
    timestamp: str
    is_winner: bool = False
    improvement_pct: float = 0.0
    error: Optional[str] = None


class LiveTracker:
    """Real-time tracking and reporting."""

    def __init__(
        self,
        output_dir: str = "results/live",
        baseline_strategy: str = "random",
        winner_threshold: float = 0.05,  # 5% improvement = winner
        webhook_url: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_strategy = baseline_strategy
        self.winner_threshold = winner_threshold
        self.webhook_url = webhook_url

        self.results: List[LiveResult] = []
        self.baseline_accuracy: Optional[float] = None
        self.start_time = datetime.now()
        self.winners: List[LiveResult] = []

        # File paths
        self.log_file = self.output_dir / "live.log"
        self.results_file = self.output_dir / "results.json"
        self.leaderboard_file = self.output_dir / "leaderboard.md"
        self.winners_file = self.output_dir / "WINNERS.md"
        self.status_file = self.output_dir / "status.json"

        # Initialize files
        self._init_files()

    def _init_files(self):
        """Initialize tracking files."""
        self._log(f"{'='*60}")
        self._log(f"MEMORYLLM BULK TEST - LIVE TRACKING")
        self._log(f"Started: {self.start_time.isoformat()}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Winner threshold: >{self.winner_threshold*100:.0f}% over baseline")
        self._log(f"{'='*60}\n")

        self._update_status("initializing", 0, 0)

    def _log(self, message: str, also_print: bool = True):
        """Write to log file and optionally print."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

        if also_print:
            print(log_line)

    def _update_status(self, phase: str, completed: int, total: int):
        """Update status file for external monitoring."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        status = {
            "phase": phase,
            "completed": completed,
            "total": total,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
            "elapsed_seconds": elapsed,
            "elapsed_human": self._format_duration(elapsed),
            "baseline_accuracy": self.baseline_accuracy,
            "num_winners": len(self.winners),
            "last_update": datetime.now().isoformat(),
            "winners": [w.strategy for w in self.winners]
        }

        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human readable."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

    def record_result(self, result: LiveResult):
        """Record a test result and update all tracking files."""
        self.results.append(result)

        # Check if this is baseline
        if result.strategy == self.baseline_strategy and not result.error:
            self.baseline_accuracy = result.accuracy
            self._log(f"📊 BASELINE SET: {result.strategy} = {result.accuracy:.2%}")

        # Check if winner
        if self.baseline_accuracy and not result.error:
            improvement = (result.accuracy - self.baseline_accuracy) / self.baseline_accuracy
            result.improvement_pct = improvement * 100

            if improvement > self.winner_threshold:
                result.is_winner = True
                self.winners.append(result)
                self._log(f"🏆 WINNER FOUND: {result.strategy} = {result.accuracy:.2%} (+{result.improvement_pct:.1f}%)")
                self._alert_winner(result)

        # Log result
        if result.error:
            self._log(f"❌ {result.strategy}: ERROR - {result.error}")
        else:
            status_icon = "🏆" if result.is_winner else "✓"
            improvement_str = f" (+{result.improvement_pct:.1f}%)" if result.improvement_pct > 0 else ""
            self._log(f"{status_icon} {result.strategy}: {result.accuracy:.2%}{improvement_str} ({result.time_seconds:.0f}s)")

        # Update all files
        self._update_results_file()
        self._update_leaderboard()
        self._update_winners_file()
        self._update_status("running", len(self.results), len(self.results))

    def _update_results_file(self):
        """Update JSON results file."""
        with open(self.results_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

    def _update_leaderboard(self):
        """Update markdown leaderboard."""
        sorted_results = sorted(
            [r for r in self.results if not r.error],
            key=lambda r: r.accuracy,
            reverse=True
        )

        md = f"# Live Leaderboard\n\n"
        md += f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += f"**Baseline ({self.baseline_strategy}):** {self.baseline_accuracy:.2%}\n\n" if self.baseline_accuracy else ""
        md += f"**Winners found:** {len(self.winners)}\n\n"
        md += "| Rank | Strategy | Accuracy | vs Baseline | Time |\n"
        md += "|------|----------|----------|-------------|------|\n"

        for i, r in enumerate(sorted_results, 1):
            icon = "🏆" if r.is_winner else ""
            improvement = f"+{r.improvement_pct:.1f}%" if r.improvement_pct > 0 else f"{r.improvement_pct:.1f}%"
            md += f"| {i} | {icon} {r.strategy} | {r.accuracy:.2%} | {improvement} | {r.time_seconds:.0f}s |\n"

        with open(self.leaderboard_file, "w") as f:
            f.write(md)

    def _update_winners_file(self):
        """Update winners markdown file."""
        if not self.winners:
            md = "# Winners\n\nNo winners found yet. Keep testing!\n"
        else:
            md = f"# 🏆 WINNERS FOUND!\n\n"
            md += f"**{len(self.winners)} strategies beat baseline by >{self.winner_threshold*100:.0f}%**\n\n"

            for i, w in enumerate(sorted(self.winners, key=lambda x: x.improvement_pct, reverse=True), 1):
                md += f"## {i}. {w.strategy}\n\n"
                md += f"- **Accuracy:** {w.accuracy:.2%}\n"
                md += f"- **Improvement:** +{w.improvement_pct:.1f}% over baseline\n"
                md += f"- **Time:** {w.time_seconds:.0f}s\n"
                md += f"- **Found at:** {w.timestamp}\n\n"

        with open(self.winners_file, "w") as f:
            f.write(md)

    def _alert_winner(self, result: LiveResult):
        """Send alert for winner (webhook if configured)."""
        message = f"🏆 WINNER: {result.strategy} = {result.accuracy:.2%} (+{result.improvement_pct:.1f}%)"

        if self.webhook_url:
            self._send_webhook(message)

    def _send_webhook(self, message: str):
        """Send webhook notification."""
        try:
            import urllib.request

            payload = json.dumps({"content": message, "text": message})
            req = urllib.request.Request(
                self.webhook_url,
                data=payload.encode(),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            self._log(f"Webhook failed: {e}", also_print=False)

    def finish(self, total_strategies: int):
        """Finalize tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        self._log(f"\n{'='*60}")
        self._log(f"BULK TEST COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"Total time: {self._format_duration(elapsed)}")
        self._log(f"Strategies tested: {len(self.results)}/{total_strategies}")
        self._log(f"Winners found: {len(self.winners)}")

        if self.winners:
            self._log(f"\n🏆 TOP WINNERS:")
            for w in sorted(self.winners, key=lambda x: x.improvement_pct, reverse=True)[:5]:
                self._log(f"   {w.strategy}: {w.accuracy:.2%} (+{w.improvement_pct:.1f}%)")

        self._update_status("complete", len(self.results), total_strategies)

        # Final summary file
        self._write_final_report()

    def _write_final_report(self):
        """Write final analysis report."""
        report_file = self.output_dir / "FINAL_REPORT.md"

        elapsed = (datetime.now() - self.start_time).total_seconds()
        sorted_results = sorted(
            [r for r in self.results if not r.error],
            key=lambda r: r.accuracy,
            reverse=True
        )

        md = f"# MemoryLLM Bulk Test Report\n\n"
        md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Duration:** {self._format_duration(elapsed)}\n"
        md += f"**Strategies Tested:** {len(self.results)}\n"
        md += f"**Baseline ({self.baseline_strategy}):** {self.baseline_accuracy:.2%}\n\n" if self.baseline_accuracy else ""

        md += f"## Results Summary\n\n"
        md += f"- **Winners (>{self.winner_threshold*100:.0f}% improvement):** {len(self.winners)}\n"
        md += f"- **Errors:** {len([r for r in self.results if r.error])}\n\n"

        if self.winners:
            md += f"## 🏆 Winners\n\n"
            for w in sorted(self.winners, key=lambda x: x.improvement_pct, reverse=True):
                md += f"- **{w.strategy}:** {w.accuracy:.2%} (+{w.improvement_pct:.1f}%)\n"
            md += "\n"

        md += f"## Full Leaderboard\n\n"
        md += "| Rank | Strategy | Accuracy | vs Baseline |\n"
        md += "|------|----------|----------|-------------|\n"
        for i, r in enumerate(sorted_results, 1):
            icon = "🏆" if r.is_winner else ""
            improvement = f"+{r.improvement_pct:.1f}%" if r.improvement_pct > 0 else f"{r.improvement_pct:.1f}%"
            md += f"| {i} | {icon} {r.strategy} | {r.accuracy:.2%} | {improvement} |\n"

        with open(report_file, "w") as f:
            f.write(md)


def watch_status(output_dir: str = "results/live", interval: int = 10):
    """Watch status file and print updates."""
    status_file = Path(output_dir) / "status.json"

    print(f"Watching {status_file}... (Ctrl+C to stop)\n")

    last_completed = -1

    while True:
        try:
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)

                if status["completed"] != last_completed:
                    last_completed = status["completed"]
                    print(f"[{status['last_update'][:19]}] "
                          f"Progress: {status['completed']}/{status['total']} "
                          f"({status['progress_pct']:.0f}%) | "
                          f"Winners: {status['num_winners']} | "
                          f"Elapsed: {status['elapsed_human']}")

                    if status['winners']:
                        print(f"   🏆 Winners: {', '.join(status['winners'])}")

                if status["phase"] == "complete":
                    print("\n✅ Test complete!")
                    break

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopped watching.")
            break
        except Exception as e:
            time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=str, help="Watch status in directory")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")

    args = parser.parse_args()

    if args.watch:
        watch_status(args.watch, args.interval)
    else:
        print("Usage: python live_tracker.py --watch results/live")
