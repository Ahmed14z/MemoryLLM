"""
Download and setup Natural Questions dataset for MemoryLLM testing.

Usage:
    python bulk_test/setup_data.py

This will create:
    data/nq/v1.0-simplified_nq-dev-all.jsonl
    data/nq/v1.0-simplified_simplified-nq-train.jsonl
    data/nq/indices_nq_4.npy
"""

import json
import numpy as np
from pathlib import Path


def setup_nq_dataset(output_dir: str = "data/nq", num_dev_samples: int = 1000):
    """Download and prepare Natural Questions dataset."""

    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dev_file = output_path / "v1.0-simplified_nq-dev-all.jsonl"
    train_file = output_path / "v1.0-simplified_simplified-nq-train.jsonl"
    indices_file = output_path / "indices_nq_4.npy"

    print("Loading Natural Questions dataset from Hugging Face...")
    print("(Streaming mode - only downloads what we need)")

    # Load dataset in streaming mode to avoid downloading 40GB
    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        streaming=True  # Stream instead of downloading everything
    )

    # Process validation set (used as dev)
    print("\nProcessing validation set...")
    valid_indices = []

    with open(dev_file, "w") as f:
        for idx, example in enumerate(dataset["validation"]):
            # Convert to expected format
            converted = convert_example(example)
            if converted is not None:
                f.write(json.dumps(converted) + "\n")
                valid_indices.append(len(valid_indices))

            if len(valid_indices) >= num_dev_samples:
                break

            if idx % 500 == 0:
                print(f"  Processed {idx} examples, kept {len(valid_indices)}")

    print(f"Saved {len(valid_indices)} dev examples to {dev_file}")

    # Process training set (for unrelated contexts)
    print("\nProcessing training set (for unrelated contexts)...")
    train_count = 0
    max_train = 5000  # We only need a subset for unrelated contexts

    with open(train_file, "w") as f:
        for idx, example in enumerate(dataset["train"]):
            converted = convert_example(example)
            if converted is not None:
                f.write(json.dumps(converted) + "\n")
                train_count += 1

            if train_count >= max_train:
                break

            if idx % 1000 == 0:
                print(f"  Processed {idx} examples, kept {train_count}")

    print(f"Saved {train_count} train examples to {train_file}")

    # Create indices file
    # This selects which samples to use for evaluation
    indices = np.arange(min(num_dev_samples, len(valid_indices)))
    np.save(indices_file, indices)
    print(f"Saved indices to {indices_file}")

    print("\nDataset setup complete!")
    print(f"Files created in {output_path}:")
    for f in output_path.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def convert_example(example: dict) -> dict:
    """Convert HuggingFace NQ format to expected jsonl format."""

    try:
        # Get annotations - in HF format these are nested with lists
        annotations = example.get("annotations", {})

        # yes_no_answer is a list of ints: -1=none, 0=no, 1=yes
        yes_no_list = annotations.get("yes_no_answer", [-1])
        yes_no = yes_no_list[0] if yes_no_list else -1

        # Skip if no valid answer
        if yes_no == -1:
            return None

        # long_answer has lists for each field
        long_answer = annotations.get("long_answer", {})
        la_start_list = long_answer.get("start_token", [-1])
        la_end_list = long_answer.get("end_token", [-1])

        la_start = la_start_list[0] if la_start_list else -1
        la_end = la_end_list[0] if la_end_list else -1

        if la_start < 0 or la_end < 0:
            return None

        # short_answers is a list of answer spans
        # Each element has start_token and end_token as lists
        short_answers_list = annotations.get("short_answers", [])

        if not short_answers_list:
            return None

        # Get first short answer
        first_short = short_answers_list[0] if short_answers_list else {}
        sa_start_list = first_short.get("start_token", [])
        sa_end_list = first_short.get("end_token", [])

        if not sa_start_list or not sa_end_list:
            return None

        sa_start = sa_start_list[0] if sa_start_list else -1
        sa_end = sa_end_list[0] if sa_end_list else -1

        if sa_start < 0 or sa_end < 0:
            return None

        # Get document tokens
        document = example.get("document", {})
        tokens = document.get("tokens", {})
        token_texts = tokens.get("token", [])
        is_html = tokens.get("is_html", [])

        if not token_texts:
            return None

        # Build document_tokens in expected format
        document_tokens = []
        for text, html in zip(token_texts, is_html):
            document_tokens.append({
                "token": text,
                "html_token": html
            })

        # Get question text
        question = example.get("question", {})
        question_text = question.get("text", "") if isinstance(question, dict) else str(question)

        # Build converted example
        converted = {
            "question_text": question_text,
            "document_tokens": document_tokens,
            "annotations": [{
                "yes_no_answer": "YES" if yes_no == 1 else "NO",
                "long_answer": {
                    "start_token": la_start,
                    "end_token": la_end
                },
                "short_answers": [{
                    "start_token": sa_start,
                    "end_token": sa_end
                }]
            }]
        }

        return converted

    except Exception as e:
        # Skip malformed examples
        return None


def verify_setup(data_dir: str = "data/nq"):
    """Verify dataset is set up correctly."""

    data_path = Path(data_dir)

    required_files = [
        "v1.0-simplified_nq-dev-all.jsonl",
        "v1.0-simplified_simplified-nq-train.jsonl",
        "indices_nq_4.npy"
    ]

    print("Verifying dataset setup...")
    all_ok = True

    for fname in required_files:
        fpath = data_path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if all_ok:
        # Try loading a sample
        try:
            import sys
            sys.path.insert(0, ".")
            from dataset.nq import NQDataset

            ds = NQDataset(
                filename=str(data_path / "v1.0-simplified_nq-dev-all.jsonl"),
                num_unrelated_contexts=2,
                num=10
            )
            print(f"\n  Dataset loads correctly! {len(ds)} samples available.")
        except Exception as e:
            print(f"\n  Warning: Dataset file exists but loading failed: {e}")

    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup NQ dataset")
    parser.add_argument("--output", type=str, default="data/nq", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Number of dev samples")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing setup")

    args = parser.parse_args()

    if args.verify_only:
        verify_setup(args.output)
    else:
        setup_nq_dataset(args.output, args.samples)
        print("\n" + "="*50)
        verify_setup(args.output)
