#!/usr/bin/env python
"""
Evaluate top-k exact match accuracy on USPTO-50K test set.
Uses pre-computed predictions from the trained NeuralSym model.
"""
import pandas as pd
from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parent / 'data'

def calculate_topk_accuracy(csv_path: str, k_values: list = None):
    """
    Calculate top-k exact match accuracy from pre-computed predictions.

    The CSV file contains a 'rank_of_true_precursor' column which is 0-indexed.
    - rank = 0 means first prediction is correct (top-1 match)
    - rank = 9999 means true precursor not found in top-200 predictions

    Args:
        csv_path: Path to the predictions CSV file
        k_values: List of k values to compute accuracy for
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50, 100, 200]

    # Load predictions
    df = pd.read_csv(csv_path)
    total = len(df)

    # Get ranks (0-indexed, 9999 means not found)
    ranks = df['rank_of_true_precursor'].values

    print("=" * 60)
    print("USPTO-50K TEST SET - EXACT MATCH ACCURACY")
    print("=" * 60)
    print(f"\nTotal test samples: {total}")
    print(f"Pre-computed predictions file: {Path(csv_path).name}")
    print("\n" + "-" * 60)
    print(f"{'Top-K':<10} {'Correct':<12} {'Accuracy':<15} {'Cumulative %'}")
    print("-" * 60)

    results = {}
    for k in k_values:
        # rank is 0-indexed, so rank < k means it's in top-k
        correct = sum(ranks < k)
        accuracy = correct / total
        results[k] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
        print(f"Top-{k:<6} {correct:<12} {accuracy*100:>6.2f}%")

    print("-" * 60)

    # Additional statistics
    not_found = sum(ranks == 9999)
    found_any = total - not_found
    print(f"\nAdditional Statistics:")
    print(f"  Found in top-200:     {found_any}/{total} ({found_any/total*100:.2f}%)")
    print(f"  Not found (rank=9999): {not_found}/{total} ({not_found/total*100:.2f}%)")

    # Mean reciprocal rank (MRR) for found predictions
    valid_ranks = ranks[ranks < 9999]
    if len(valid_ranks) > 0:
        mrr = sum(1.0 / (r + 1) for r in valid_ranks) / total
        print(f"  Mean Reciprocal Rank: {mrr:.4f}")

    print("=" * 60)

    return results


def main():
    # Path to pre-computed test predictions
    test_csv = DATA_FOLDER / 'proposal_original' / 'neuralsym_200topk_200maxk_noGT_test.csv'

    if not test_csv.exists():
        print(f"Error: Predictions file not found at {test_csv}")
        print("Please ensure the pre-computed predictions exist.")
        return

    # Calculate accuracy for various k values, highlighting top-10
    print("\n" + "=" * 60)
    print("NEURALSYM MODEL EVALUATION")
    print("=" * 60)

    results = calculate_topk_accuracy(
        test_csv,
        k_values=[1, 2, 3, 5, 10, 20, 50, 100, 200]
    )

    # Highlight the requested top-10 accuracy
    print("\n" + "=" * 60)
    print("REQUESTED METRIC: TOP-10 EXACT MATCH ACCURACY")
    print("=" * 60)
    top10 = results[10]
    print(f"\n  >>> Top-10 Accuracy: {top10['accuracy']*100:.2f}% ({top10['correct']}/{top10['total']})")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
