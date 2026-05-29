from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs train size for labeled reducing-data results.")
    parser.add_argument("--results_csv", type=str, default="results/labeled_reducing_data_32/results.csv")
    parser.add_argument("--output_path", type=str, default="results/labeled_reducing_data_32/accuracy_vs_train_size.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV does not exist: {results_csv}. Run scripts/dev/run_labeled_reducing_data.py first."
        )

    rows = _read_rows(results_csv)
    if not rows:
        raise RuntimeError(f"No rows found in results csv: {args.results_csv}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    by_method = _group_by_method(rows)
    for method, method_rows in sorted(by_method.items()):
        sorted_rows = sorted(method_rows, key=lambda row: int(row["train_size"]))
        x_values = [int(row["train_size"]) for row in sorted_rows]
        y_values = [float(row["accuracy"]) for row in sorted_rows]
        plt.plot(x_values, y_values, marker="o", label=method)

    plt.title("Labeled reducing data debug")
    plt.xlabel("train_size")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"saved plot: {output_path}")


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_by_method(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return grouped


if __name__ == "__main__":
    main()
