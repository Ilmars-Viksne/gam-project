from pathlib import Path

import pandas as pd


main_results = pd.read_csv(
    Path("outputs")
    / "nested_cv_fold_results.csv"
)

pairwise_results = pd.read_csv(
    Path("outputs_pairwise")
    / "nested_cv_fold_results.csv"
)

comparison = main_results.merge(
    pairwise_results,
    on=["repeat", "fold"],
    suffixes=("_main", "_pairwise"),
    validate="one_to_one",
)

comparison["log_loss_difference"] = (
    comparison["log_loss_pairwise"]
    - comparison["log_loss_main"]
)

comparison["accuracy_difference"] = (
    comparison["accuracy_pairwise"]
    - comparison["accuracy_main"]
)

comparison["balanced_accuracy_difference"] = (
    comparison["balanced_accuracy_pairwise"]
    - comparison["balanced_accuracy_main"]
)

comparison["macro_f1_difference"] = (
    comparison["macro_f1_pairwise"]
    - comparison["macro_f1_main"]
)

print(
    comparison[
        [
            "log_loss_difference",
            "accuracy_difference",
            "balanced_accuracy_difference",
            "macro_f1_difference",
        ]
    ]
    .agg(["mean", "std", "median"])
    .T
)

comparison.to_csv(
    "outputs_pairwise/"
    "main_vs_pairwise_comparison.csv",
    index=False,
)