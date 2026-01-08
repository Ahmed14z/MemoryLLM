"""
Parallel bulk testing runner for multi-GPU setups.

Runs multiple strategy tests in parallel across GPUs.

Usage:
    # Auto-detect GPUs and run in parallel
    python bulk_test/parallel_runner.py --all --samples 200 --nuc 20

    # Specify number of parallel workers
    python bulk_test/parallel_runner.py --all --workers 4
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from bulk_test.drop_strategies import list_strategies
from bulk_test.live_tracker import LiveTracker, LiveResult


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_info_all() -> List[dict]:
    """Get info for all GPUs."""
    gpus = []
    try:
        import torch
        for i in range(get_num_gpus()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_gb": props.total_memory / 1024**3
            })
    except ImportError:
        pass
    return gpus


def create_test_script(strategy: str, gpu_id: int, model_path: str,
                       num_samples: int, num_contexts: int) -> str:
    """Create a test script as a string."""
    script = '''import sys
sys.path.insert(0, '.')
import os
import gc
# CUDA_VISIBLE_DEVICES is set via subprocess env, not here

import torch
import json
from datetime import datetime

# Clear any leftover GPU memory
torch.cuda.empty_cache()
gc.collect()

strategy = "{strategy}"
model_path = "{model_path}"
num_samples = {num_samples}
num_contexts = {num_contexts}
gpu_id = {gpu_id}

try:
    print(f"=== DEBUG INFO ===")
    print(f"CUDA_VISIBLE_DEVICES: {{os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}}")
    print(f"torch.cuda.device_count(): {{torch.cuda.device_count()}}")
    print(f"torch.cuda.current_device(): {{torch.cuda.current_device()}}")
    print(f"GPU memory before import: {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

    from modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer

    print(f"Loading model on GPU {{gpu_id}} with strategy: {{strategy}}")
    print(f"GPU memory after import: {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

    # Load model with explicit device mapping to avoid loading on wrong GPU
    print(f"Loading model...")
    model = MemoryLLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    print(f"GPU memory after from_pretrained: {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

    model = model.cuda()
    print(f"GPU memory after .cuda(): {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")

    # Print model memory config
    print(f"Model config: num_tokens={{model.num_tokens}}, num_blocks={{model.num_blocks}}")
    print(f"Memory shape: {{model.memory.shape}}")
    print(f"Memory size: {{model.memory.numel() * 2 / 1e9:.2f}} GB (bf16)")

    # Set drop strategy after loading
    model.drop_strategy = strategy
    if strategy != 'random':
        from bulk_test.drop_strategies import MemoryTracker
        model.memory_tracker = MemoryTracker(
            num_layers=model.config.num_hidden_layers,
            num_tokens=model.num_blocks * model.num_tokens,
            device='cuda'
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    from dataset.nq import NQDataset
    dataset = NQDataset(
        filename="./data/nq/v1.0-simplified_nq-dev-all.jsonl",
        num_unrelated_contexts=num_contexts,
        tokenizer='llama',
        tokenizer_path=model_path
    )

    # Run evaluation
    correct = 0
    total = min(num_samples, len(dataset))

    for i in range(total):
        context, question, answer, unrelated_contexts = dataset[i]

        # Inject memories
        context_ids = tokenizer(context, return_tensors="pt").input_ids.cuda()
        model.inject_memory(context_ids, update_memory=True)

        for uc in unrelated_contexts[:num_contexts]:
            uc_ids = tokenizer(uc, return_tensors="pt").input_ids.cuda()
            model.inject_memory(uc_ids, update_memory=True)

        # Generate
        question_encoded = tokenizer(question, return_tensors="pt")
        question_ids = question_encoded.input_ids.cuda()
        attention_mask = torch.cat([
            torch.ones(1, model.num_tokens * (model.num_blocks - 1)).cuda(),
            question_encoded.attention_mask.cuda()
        ], dim=1)

        with torch.no_grad():
            output = model.generate(
                inputs=question_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        if answer.lower() in prediction.lower():
            correct += 1

        # Reset memory
        model.initialized.fill_(0)

        if (i + 1) % 20 == 0:
            acc_so_far = correct / (i + 1) * 100
            print(f"  Progress: {{i+1}}/{{total}} ({{acc_so_far:.1f}}% acc so far)")

    accuracy = correct / total if total > 0 else 0.0

    result = {{
        "strategy": strategy,
        "accuracy": accuracy,
        "num_samples": total,
        "num_contexts": num_contexts,
        "gpu_id": gpu_id,
        "error": None
    }}

except Exception as e:
    result = {{
        "strategy": strategy,
        "accuracy": 0.0,
        "num_samples": 0,
        "num_contexts": num_contexts,
        "gpu_id": gpu_id,
        "error": str(e)
    }}

print(f"RESULT_JSON:{{json.dumps(result)}}")
'''.format(
        strategy=strategy,
        gpu_id=gpu_id,
        model_path=model_path,
        num_samples=num_samples,
        num_contexts=num_contexts
    )
    return script


def run_strategy_on_gpu(
    strategy: str,
    gpu_id: int,
    model_path: str,
    num_samples: int,
    num_contexts: int,
    output_dir: str
) -> dict:
    """Run a single strategy test on a specific GPU."""

    start_time = time.time()

    # Create test script
    test_script = create_test_script(strategy, gpu_id, model_path, num_samples, num_contexts)

    # Write temp script
    script_path = Path(output_dir) / f"_temp_{strategy}_{gpu_id}.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(test_script)

    try:
        # Run the test with CUDA_VISIBLE_DEVICES set in environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(Path(__file__).parent.parent),
            env=env
        )

        # Parse result from output
        output = result.stdout + result.stderr

        # Print full output for debugging if error occurred
        if result.returncode != 0 or 'Error' in output or 'error' in output.lower():
            print(f"\n=== FULL OUTPUT FOR {strategy} (GPU {gpu_id}) ===")
            print(output[-3000:])  # Last 3000 chars
            print("=== END OUTPUT ===\n")

        # Find JSON result
        for line in output.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result_json = json.loads(line[12:])
                result_json['time_seconds'] = time.time() - start_time
                result_json['timestamp'] = datetime.now().isoformat()
                return result_json

        # Fallback if no result found
        return {
            "strategy": strategy,
            "accuracy": 0.0,
            "time_seconds": time.time() - start_time,
            "num_samples": 0,
            "num_contexts": num_contexts,
            "gpu_id": gpu_id,
            "error": "No result found in output: " + output[-500:],
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        return {
            "strategy": strategy,
            "accuracy": 0.0,
            "time_seconds": time.time() - start_time,
            "num_samples": 0,
            "num_contexts": num_contexts,
            "gpu_id": gpu_id,
            "error": "Timeout",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "strategy": strategy,
            "accuracy": 0.0,
            "time_seconds": time.time() - start_time,
            "num_samples": 0,
            "num_contexts": num_contexts,
            "gpu_id": gpu_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # Cleanup temp script
        if script_path.exists():
            script_path.unlink()


def run_parallel_tests(
    strategies: List[str],
    num_workers: int = None,
    num_samples: int = 200,
    num_contexts: int = 20,
    output_dir: str = "results/parallel",
    model_path: str = "YuWangX/memoryllm-8b",
    webhook_url: str = None
):
    """Run tests in parallel across GPUs."""

    # Auto-detect workers
    num_gpus = get_num_gpus()
    if num_workers is None:
        num_workers = max(1, num_gpus)

    # Ensure random is first
    if "random" in strategies:
        strategies.remove("random")
    strategies.insert(0, "random")

    # Initialize tracker
    tracker = LiveTracker(
        output_dir=output_dir,
        baseline_strategy="random",
        winner_threshold=0.05,
        webhook_url=webhook_url
    )

    print("\n" + "="*60)
    print("MEMORYLLM PARALLEL BULK TESTING")
    print("="*60)
    print(f"GPUs detected: {num_gpus}")
    for gpu in get_gpu_info_all():
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.0f}GB)")
    print(f"Parallel workers: {num_workers}")
    print(f"Strategies to test: {len(strategies)}")
    print(f"Samples per test: {num_samples}")
    print(f"Unrelated contexts: {num_contexts}")
    est_time = len(strategies) / num_workers * 15
    seq_time = len(strategies) * 15
    print(f"\nEstimated time: ~{est_time:.0f} minutes")
    print(f"(vs ~{seq_time:.0f} minutes sequential)")
    print("="*60 + "\n")

    # Assign strategies to GPUs round-robin
    gpu_assignments = []
    for i, strategy in enumerate(strategies):
        assigned_gpu = i % num_workers
        gpu_assignments.append((strategy, assigned_gpu))

    # Run in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {}
        for strategy, assigned_gpu in gpu_assignments:
            future = executor.submit(
                run_strategy_on_gpu,
                strategy=strategy,
                gpu_id=assigned_gpu,
                model_path=model_path,
                num_samples=num_samples,
                num_contexts=num_contexts,
                output_dir=output_dir
            )
            futures[future] = strategy

        # Process as they complete
        for future in as_completed(futures):
            strategy = futures[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                # Record to tracker
                live_result = LiveResult(
                    strategy=result['strategy'],
                    accuracy=result['accuracy'],
                    time_seconds=result.get('time_seconds', 0),
                    num_contexts=result['num_contexts'],
                    num_samples=result['num_samples'],
                    timestamp=result.get('timestamp', ''),
                    error=result.get('error')
                )
                tracker.record_result(live_result)

                # Print status
                total_strategies = len(strategies)
                if result.get('error'):
                    status = "ERROR"
                else:
                    status = f"{result['accuracy']:.2%}"
                gpu_used = result.get('gpu_id', '?')
                print(f"[{completed}/{total_strategies}] {strategy}: {status} (GPU {gpu_used})")

            except Exception as e:
                total_strategies = len(strategies)
                print(f"[{completed}/{total_strategies}] {strategy}: FAILED - {e}")

    # Finalize
    tracker.finish(len(strategies))

    print("\nAll tests complete!")
    print(f"Results: {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel bulk testing")

    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategies")
    parser.add_argument("--all", action="store_true",
                        help="Test all strategies")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                        help="Test specific tier")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Samples per test")
    parser.add_argument("--nuc", type=int, default=20,
                        help="Unrelated contexts")
    parser.add_argument("--output", type=str, default="results/parallel",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="YuWangX/memoryllm-8b",
                        help="Model path")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Webhook URL for alerts")
    parser.add_argument("--list-gpus", action="store_true",
                        help="List GPUs and exit")

    args = parser.parse_args()

    if args.list_gpus:
        num_gpus = get_num_gpus()
        print(f"GPUs: {num_gpus}")
        for gpu in get_gpu_info_all():
            print(f"  {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.0f}GB)")
        return

    # Determine strategies
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
        strategies = ["random", "fifo", "lru", "norm_low"]

    run_parallel_tests(
        strategies=strategies,
        num_workers=args.workers,
        num_samples=args.samples,
        num_contexts=args.nuc,
        output_dir=args.output,
        model_path=args.model,
        webhook_url=args.webhook
    )


if __name__ == "__main__":
    main()
