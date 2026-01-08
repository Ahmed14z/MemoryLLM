"""
Bulk testing runner for MemoryLLM drop strategies.

Usage:
    python bulk_test/runner.py --strategies random,fifo,lru --samples 100 --nuc 10
    python bulk_test/runner.py --all --output results/bulk_test
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bulk_test.drop_strategies import STRATEGIES, list_strategies


@dataclass
class TestConfig:
    """Configuration for a single test run."""
    strategy: str
    num_unrelated_contexts: int = 10
    num_samples: int = 100
    datasets: List[str] = None
    model_path: str = "YuWangX/memoryllm-7b"

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["naturalqa"]


@dataclass
class TestResult:
    """Results from a single test run."""
    strategy: str
    accuracy: float
    time_seconds: float
    num_contexts: int
    num_samples: int
    gpu_memory_mb: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "memory_gb": 0}

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "memory_used_gb": torch.cuda.memory_allocated() / 1024**3
    }


def detect_optimal_settings() -> dict:
    """Auto-detect optimal settings based on GPU."""
    gpu = get_gpu_info()

    if not gpu["available"]:
        return {
            "model": "YuWangX/memoryllm-7b",
            "quantization": "4bit",
            "batch_size": 1,
            "num_samples": 50
        }

    vram = gpu["memory_gb"]

    if vram >= 70:  # H100, A100-80GB
        return {
            "model": "YuWangX/memoryllm-8b",
            "quantization": "none",
            "batch_size": 8,
            "num_samples": 200
        }
    elif vram >= 40:  # A100-40GB, A6000
        return {
            "model": "YuWangX/memoryllm-8b",
            "quantization": "none",
            "batch_size": 4,
            "num_samples": 150
        }
    elif vram >= 20:  # RTX 3090, 4090
        return {
            "model": "YuWangX/memoryllm-7b",
            "quantization": "8bit",
            "batch_size": 2,
            "num_samples": 100
        }
    else:  # RTX 4060 Ti 16GB, etc
        return {
            "model": "YuWangX/memoryllm-7b",
            "quantization": "4bit",
            "batch_size": 1,
            "num_samples": 50
        }


def run_single_test_inline(config: TestConfig) -> TestResult:
    """
    Run a single test inline (in same process).
    More reliable than subprocess but uses more memory.
    """
    start_time = time.time()

    try:
        # Import here to avoid loading model at import time
        from configuration_memoryllm import MemoryLLMConfig
        from modeling_memoryllm import MemoryLLM
        from transformers import AutoTokenizer, BitsAndBytesConfig

        print(f"  Loading model with strategy: {config.strategy}")

        # Detect settings
        settings = detect_optimal_settings()

        # Create config with strategy
        model_config = MemoryLLMConfig.from_pretrained(
            config.model_path,
            drop_strategy=config.strategy
        )
        model_config.drop_strategy = config.strategy

        # Load model
        if settings["quantization"] == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4'
            )
            model = MemoryLLM.from_pretrained(
                config.model_path,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = MemoryLLM.from_pretrained(
                config.model_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # Load test data
        from dataset.nq import NQDataset
        dataset = NQDataset(
            filename="./data/nq/v1.0-simplified_nq-dev-all.jsonl",
            num_unrelated_contexts=config.num_unrelated_contexts,
            tokenizer='llama',
            tokenizer_path=config.model_path
        )

        # Run evaluation
        correct = 0
        total = min(config.num_samples, len(dataset))

        for i in range(total):
            context, question, answer, unrelated_contexts = dataset[i]

            # Inject context
            context_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
            model.inject_memory(context_ids, update_memory=True)

            # Inject unrelated contexts
            for uc in unrelated_contexts[:config.num_unrelated_contexts]:
                uc_ids = tokenizer(uc, return_tensors="pt").input_ids.to(model.device)
                model.inject_memory(uc_ids, update_memory=True)

            # Generate answer
            question_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
            output = model.generate(question_ids, max_new_tokens=50)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            # Check accuracy
            if answer.lower() in prediction.lower():
                correct += 1

            # Reset memory for next sample
            model.initialized.fill_(0)

        accuracy = correct / total if total > 0 else 0.0
        gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        return TestResult(
            strategy=config.strategy,
            accuracy=accuracy,
            time_seconds=time.time() - start_time,
            num_contexts=config.num_unrelated_contexts,
            num_samples=total,
            gpu_memory_mb=gpu_mem
        )

    except Exception as e:
        return TestResult(
            strategy=config.strategy,
            accuracy=0.0,
            time_seconds=time.time() - start_time,
            num_contexts=config.num_unrelated_contexts,
            num_samples=0,
            error=str(e)
        )


def run_single_test_subprocess(config: TestConfig, script_path: str = "test_qa_memory.py") -> TestResult:
    """
    Run a single test via subprocess.
    Cleaner memory management but requires parsing output.
    """
    start_time = time.time()

    # Set environment variable for strategy
    env = os.environ.copy()
    env["MEMORYLLM_DROP_STRATEGY"] = config.strategy

    cmd = [
        sys.executable, script_path,
        "--model", config.model_path,
        "--nuc", str(config.num_unrelated_contexts),
        "--datasets", *config.datasets,
        "--num_samples", str(config.num_samples),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env,
            cwd=str(Path(__file__).parent.parent)
        )

        # Parse accuracy from output
        accuracy = parse_accuracy_from_output(result.stdout)

        return TestResult(
            strategy=config.strategy,
            accuracy=accuracy,
            time_seconds=time.time() - start_time,
            num_contexts=config.num_unrelated_contexts,
            num_samples=config.num_samples,
            error=result.stderr if result.returncode != 0 else None
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            strategy=config.strategy,
            accuracy=0.0,
            time_seconds=time.time() - start_time,
            num_contexts=config.num_unrelated_contexts,
            num_samples=config.num_samples,
            error="Timeout"
        )
    except Exception as e:
        return TestResult(
            strategy=config.strategy,
            accuracy=0.0,
            time_seconds=time.time() - start_time,
            num_contexts=config.num_unrelated_contexts,
            num_samples=config.num_samples,
            error=str(e)
        )


def parse_accuracy_from_output(output: str) -> float:
    """Parse accuracy from test script output."""
    import re

    # Look for patterns like "accuracy: 0.85" or "Accuracy: 85%"
    patterns = [
        r"accuracy[:\s]+([0-9.]+)",
        r"acc[:\s]+([0-9.]+)",
        r"exact.?hit[:\s]+([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output.lower())
        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            if value > 1:
                value /= 100
            return value

    return 0.0


def run_bulk_tests(
    strategies: List[str],
    num_samples: int = 100,
    num_contexts: int = 10,
    output_dir: str = "results/bulk",
    model_path: str = None
) -> List[TestResult]:
    """Run tests for multiple strategies."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect settings
    settings = detect_optimal_settings()
    if model_path is None:
        model_path = settings["model"]

    print("\n" + "="*60)
    print("MEMORYLLM BULK TESTING")
    print("="*60)
    print(f"GPU: {get_gpu_info()}")
    print(f"Model: {model_path}")
    print(f"Strategies: {len(strategies)}")
    print(f"Samples per test: {num_samples}")
    print(f"Unrelated contexts: {num_contexts}")
    print("="*60 + "\n")

    results = []

    for i, strategy in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] Testing: {strategy}")
        print("-" * 40)

        config = TestConfig(
            strategy=strategy,
            num_unrelated_contexts=num_contexts,
            num_samples=num_samples,
            model_path=model_path
        )

        # Use subprocess for cleaner memory management
        result = run_single_test_subprocess(config)
        results.append(result)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Accuracy: {result.accuracy:.2%}")
            print(f"  Time: {result.time_seconds:.1f}s")

        # Save incrementally
        save_results(results, output_path)

    # Final summary
    print_summary(results)

    return results


def save_results(results: List[TestResult], output_path: Path):
    """Save results to JSON file."""
    results_file = output_path / "results.json"

    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Also save as CSV for easy viewing
    csv_file = output_path / "results.csv"
    with open(csv_file, "w") as f:
        f.write("strategy,accuracy,time_seconds,num_contexts,num_samples,error\n")
        for r in results:
            error = r.error.replace(",", ";") if r.error else ""
            f.write(f"{r.strategy},{r.accuracy:.4f},{r.time_seconds:.1f},{r.num_contexts},{r.num_samples},{error}\n")


def print_summary(results: List[TestResult]):
    """Print summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Sort by accuracy
    sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)

    print("\nLeaderboard:")
    print("-" * 50)
    for i, r in enumerate(sorted_results, 1):
        status = "ERROR" if r.error else f"{r.accuracy:.2%}"
        print(f"  {i:2}. {r.strategy:25} {status:>10}")

    # Find best non-random
    baseline = next((r for r in results if r.strategy == "random"), None)
    if baseline and not baseline.error:
        print(f"\nBaseline (random): {baseline.accuracy:.2%}")

        winners = [r for r in sorted_results
                   if r.accuracy > baseline.accuracy * 1.05 and not r.error]
        if winners:
            print(f"\nStrategies beating baseline by >5%:")
            for r in winners:
                improvement = (r.accuracy - baseline.accuracy) / baseline.accuracy * 100
                print(f"  - {r.strategy}: +{improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Bulk test MemoryLLM drop strategies")

    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated list of strategies to test")
    parser.add_argument("--all", action="store_true",
                        help="Test all available strategies")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Test strategies from specific tier")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per test")
    parser.add_argument("--nuc", type=int, default=10,
                        help="Number of unrelated contexts")
    parser.add_argument("--output", type=str, default="results/bulk",
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (auto-detected if not specified)")
    parser.add_argument("--list", action="store_true",
                        help="List available strategies and exit")

    args = parser.parse_args()

    if args.list:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        return

    # Determine which strategies to test
    if args.all:
        strategies = list_strategies()
    elif args.tier:
        tier_strategies = {
            1: ["random", "fifo", "lifo", "lru", "lfu", "mru", "round_robin"],
            2: ["norm_low", "norm_high", "variance_low", "entropy", "cosine_similar", "cosine_dissimilar"],
            3: ["hybrid_lru_random", "hybrid_fifo_importance", "tiered", "diversity", "probabilistic_lru", "adaptive"],
            4: ["sliding_landmarks", "exponential_decay", "temperature", "two_tier", "layer_aware", "attention_score"]
        }
        strategies = tier_strategies.get(args.tier, [])
    elif args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        # Default: basic strategies
        strategies = ["random", "fifo", "lru", "norm_low"]

    # Validate strategies
    available = set(list_strategies())
    invalid = [s for s in strategies if s not in available]
    if invalid:
        print(f"Warning: Unknown strategies ignored: {invalid}")
        strategies = [s for s in strategies if s in available]

    if not strategies:
        print("No valid strategies to test!")
        return

    # Run tests
    results = run_bulk_tests(
        strategies=strategies,
        num_samples=args.samples,
        num_contexts=args.nuc,
        output_dir=args.output,
        model_path=args.model
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
