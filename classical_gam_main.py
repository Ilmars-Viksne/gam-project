from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class GAMConfig:
    data_path: Path = Path("data/dataset.csv")
    output_directory: Path = Path("outputs")

    target_column: str = "Y"

    # Predictors with enough resolution to support smooth functions.
    smooth_features: tuple[str, ...] = (
        "X1",
        "X2",
        "X4",
        "X5",
        "X7",
    )

    # These variables contain only a few distinct observed values.
    categorical_features: tuple[str, ...] = (
        "X3",
        "X6",
    )

    class_order: tuple[str, ...] = (
        "O",
        "B",
        "M",
        "G",
    )

    random_state: int = 42

    # Outer CV estimates generalization performance.
    outer_splits: int = 5
    outer_repeats: int = 5

    # Inner CV tunes spline complexity and regularization.
    inner_splits: int = 5

    # Small grids are appropriate for this moderate-size dataset.
    n_knots_grid: tuple[int, ...] = (3, 4, 5, 6)
    degree_grid: tuple[int, ...] = (2, 3)
    c_grid: tuple[float, ...] = (
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    )


# ============================================================
# Classical multinomial GAM
# ============================================================


class ClassicalMultinomialGAM:
    """
    Classical main-effects multinomial GAM implemented as:

        univariate B-spline bases
        + multinomial logistic regression
        + L2 coefficient regularization

    No interaction features are constructed.
    """

    def __init__(self, config: GAMConfig) -> None:
        self.config = config

        self.data: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None

        self.grid_search: GridSearchCV | None = None
        self.best_model: Pipeline | None = None

    # --------------------------------------------------------
    # Data loading and audit
    # --------------------------------------------------------

    def load_data(self) -> None:
        config = self.config

        df = pd.read_csv(config.data_path)

        required_columns = [
            *config.smooth_features,
            *config.categorical_features,
            config.target_column,
        ]

        missing_columns = sorted(set(required_columns) - set(df.columns))

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        df = df.loc[:, required_columns].copy()

        numeric_features = [
            *config.smooth_features,
            *config.categorical_features,
        ]

        for column in numeric_features:
            df[column] = pd.to_numeric(df[column], errors="raise")

        if df[required_columns].isna().any().any():
            missing_counts = df[required_columns].isna().sum()
            raise ValueError(
                "Missing values were found:\n"
                f"{missing_counts[missing_counts > 0]}"
            )

        observed_classes = set(df[config.target_column].unique())
        expected_classes = set(config.class_order)

        unexpected_classes = observed_classes - expected_classes

        if unexpected_classes:
            raise ValueError(
                f"Unexpected target classes: {unexpected_classes}"
            )

        df[config.target_column] = pd.Categorical(
            df[config.target_column],
            categories=config.class_order,
            ordered=False,
        )

        # Convert low-resolution variables to strings so the
        # one-hot encoder treats them as categorical states.
        for column in config.categorical_features:
            df[column] = df[column].astype(str)

        self.data = df
        self.X = df.drop(columns=config.target_column)
        self.y = df[config.target_column].astype(str)

        config.output_directory.mkdir(parents=True, exist_ok=True)

    def audit_data(self) -> dict[str, Any]:
        self._require_loaded_data()

        assert self.data is not None
        assert self.X is not None
        assert self.y is not None

        config = self.config

        numeric_view = self.data[
            list(config.smooth_features)
        ].copy()

        class_counts = self.y.value_counts().reindex(
            config.class_order,
            fill_value=0,
        )

        class_proportions = (
            class_counts / class_counts.sum()
        )

        unique_counts = self.X.nunique()

        exact_duplicate_count = int(
            self.X.duplicated(keep=False).sum()
        )

        duplicate_groups = (
            self.data
            .groupby(
                list(self.X.columns),
                observed=True,
                dropna=False,
            )[config.target_column]
            .agg(
                number_of_rows="size",
                number_of_labels="nunique",
                labels=lambda values: ",".join(
                    sorted(set(values.astype(str)))
                ),
            )
            .reset_index()
        )

        conflicting_duplicates = duplicate_groups.loc[
            (duplicate_groups["number_of_rows"] > 1)
            & (duplicate_groups["number_of_labels"] > 1)
        ]

        x5_unusual = self.data.loc[
            self.data["X5"] > 40
        ].copy()

        correlation_matrix = numeric_view.corr()

        audit = {
            "number_of_rows": int(len(self.data)),
            "class_counts": class_counts.to_dict(),
            "class_proportions": class_proportions.to_dict(),
            "unique_values": unique_counts.to_dict(),
            "rows_in_duplicate_configurations": exact_duplicate_count,
            "conflicting_duplicate_groups": int(
                len(conflicting_duplicates)
            ),
            "observations_with_x5_above_40": int(
                len(x5_unusual)
            ),
        }

        pd.DataFrame(
            {
                "class": class_counts.index,
                "count": class_counts.values,
                "proportion": class_proportions.values,
            }
        ).to_csv(
            config.output_directory / "class_distribution.csv",
            index=False,
        )

        correlation_matrix.to_csv(
            config.output_directory / "correlation_matrix.csv"
        )

        duplicate_groups.to_csv(
            config.output_directory / "duplicate_groups.csv",
            index=False,
        )

        conflicting_duplicates.to_csv(
            config.output_directory
            / "conflicting_duplicate_groups.csv",
            index=False,
        )

        x5_unusual.to_csv(
            config.output_directory / "unusual_x5_observations.csv",
            index=False,
        )

        with (
            config.output_directory / "data_audit.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2)

        return audit

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------

    def build_pipeline(self) -> Pipeline:
        config = self.config

        spline_transformer = SplineTransformer(
            n_knots=5,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        )

        categorical_transformer = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "smooth",
                    spline_transformer,
                    list(config.smooth_features),
                ),
                (
                    "categorical",
                    categorical_transformer,
                    list(config.categorical_features),
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        )

        classifier = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=10_000,
            class_weight=None,
            random_state=config.random_state,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def parameter_grid(self) -> dict[str, list[Any]]:
        config = self.config

        return {
            "preprocessor__smooth__n_knots": list(
                config.n_knots_grid
            ),
            "preprocessor__smooth__degree": list(
                config.degree_grid
            ),
            "classifier__C": list(config.c_grid),
        }

    # --------------------------------------------------------
    # Nested cross-validation
    # --------------------------------------------------------

    def evaluate_nested_cv(self) -> pd.DataFrame:
        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config
        base_pipeline = self.build_pipeline()

        inner_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.random_state,
        )

        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        outer_cv = RepeatedStratifiedKFold(
            n_splits=config.outer_splits,
            n_repeats=config.outer_repeats,
            random_state=config.random_state,
        )

        scoring = {
            "log_loss": "neg_log_loss",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        }

        results = cross_validate(
            estimator=search,
            X=self.X,
            y=self.y,
            scoring=scoring,
            cv=outer_cv,
            n_jobs=1,
            return_estimator=True,
            return_train_score=False,
            error_score="raise",
        )

        fold_results = pd.DataFrame(
            {
                "fold": np.arange(
                    1,
                    len(results["test_accuracy"]) + 1,
                ),
                "log_loss": -results["test_log_loss"],
                "accuracy": results["test_accuracy"],
                "balanced_accuracy": (
                    results["test_balanced_accuracy"]
                ),
                "macro_f1": results["test_macro_f1"],
                "fit_time_seconds": results["fit_time"],
                "score_time_seconds": results["score_time"],
            }
        )

        best_parameters = [
            estimator.best_params_
            for estimator in results["estimator"]
        ]

        fold_results["best_n_knots"] = [
            parameters[
                "preprocessor__smooth__n_knots"
            ]
            for parameters in best_parameters
        ]

        fold_results["best_degree"] = [
            parameters[
                "preprocessor__smooth__degree"
            ]
            for parameters in best_parameters
        ]

        fold_results["best_C"] = [
            parameters["classifier__C"]
            for parameters in best_parameters
        ]

        fold_results.to_csv(
            config.output_directory
            / "nested_cv_fold_results.csv",
            index=False,
        )

        summary = (
            fold_results[
                [
                    "log_loss",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                ]
            ]
            .agg(["mean", "std", "min", "max"])
            .T
        )

        summary["standard_error"] = (
            summary["std"] / np.sqrt(len(fold_results))
        )

        summary.to_csv(
            config.output_directory
            / "nested_cv_summary.csv"
        )

        return fold_results

    # --------------------------------------------------------
    # Final full-data model
    # --------------------------------------------------------

    def fit_final_model(self) -> GridSearchCV:
        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config

        inner_cv = StratifiedKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.random_state,
        )

        self.grid_search = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise",
        )

        self.grid_search.fit(self.X, self.y)
        self.best_model = self.grid_search.best_estimator_

        joblib.dump(
            self.best_model,
            config.output_directory
            / "classical_gam_main_effects.joblib",
        )

        best_parameters = {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in self.grid_search.best_params_.items()
        }

        with (
            config.output_directory
            / "best_hyperparameters.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(best_parameters, file, indent=2)

        return self.grid_search

    # --------------------------------------------------------
    # Descriptive full-data diagnostics
    # --------------------------------------------------------

    def full_data_diagnostics(self) -> dict[str, Any]:
        self._require_fitted_model()

        assert self.best_model is not None
        assert self.X is not None
        assert self.y is not None

        config = self.config

        probabilities = self.best_model.predict_proba(self.X)
        predictions = self.best_model.predict(self.X)

        classifier = self.best_model.named_steps["classifier"]
        class_labels = classifier.classes_

        row_sum_error = float(
            np.abs(probabilities.sum(axis=1) - 1.0).max()
        )

        if row_sum_error > 1e-10:
            raise RuntimeError(
                "Predicted class probabilities do not sum to one."
            )

        metrics = {
            "accuracy": float(
                accuracy_score(self.y, predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    self.y,
                    predictions,
                )
            ),
            "macro_f1": float(
                f1_score(
                    self.y,
                    predictions,
                    average="macro",
                )
            ),
            "multiclass_log_loss": float(
                log_loss(
                    self.y,
                    probabilities,
                    labels=class_labels,
                )
            ),
            "maximum_probability_sum_error": row_sum_error,
        }

        with (
            config.output_directory
            / "full_data_descriptive_metrics.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        report = classification_report(
            self.y,
            predictions,
            labels=list(class_labels),
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).T.to_csv(
            config.output_directory
            / "full_data_classification_report.csv"
        )

        matrix = confusion_matrix(
            self.y,
            predictions,
            labels=class_labels,
        )

        pd.DataFrame(
            matrix,
            index=[f"Observed_{x}" for x in class_labels],
            columns=[
                f"Predicted_{x}" for x in class_labels
            ],
        ).to_csv(
            config.output_directory
            / "full_data_confusion_matrix.csv"
        )

        prediction_output = self.X.copy()
        prediction_output["observed_class"] = self.y.values
        prediction_output["predicted_class"] = predictions

        for class_index, class_name in enumerate(class_labels):
            prediction_output[
                f"probability_{class_name}"
            ] = probabilities[:, class_index]

        prediction_output["correct"] = (
            prediction_output["observed_class"]
            == prediction_output["predicted_class"]
        )

        prediction_output.to_csv(
            config.output_directory
            / "full_data_fitted_predictions.csv",
            index=False,
        )

        self._plot_confusion_matrix(
            matrix=matrix,
            class_labels=class_labels,
        )

        return metrics

    # --------------------------------------------------------
    # Main-effect curves
    # --------------------------------------------------------

    def plot_main_effects(
        self,
        grid_size: int = 200,
        central_quantile_range: tuple[float, float] = (
            0.01,
            0.99,
        ),
    ) -> None:
        """
        Plot class-specific additive score contributions.

        For each continuous feature, the plot shows:

            spline_basis(x) @ corresponding_coefficients

        with the contribution centered using the empirical mean
        over the observed feature values.

        These are additive logit-score contributions, not direct
        changes in class probabilities.
        """

        self._require_fitted_model()

        assert self.best_model is not None
        assert self.X is not None

        config = self.config

        preprocessor = self.best_model.named_steps[
            "preprocessor"
        ]
        classifier = self.best_model.named_steps[
            "classifier"
        ]

        fitted_spline_transformer = (
            preprocessor.named_transformers_["smooth"]
        )

        n_splines_per_feature = (
            fitted_spline_transformer.n_features_out_
            // len(config.smooth_features)
        )

        coefficients = classifier.coef_
        class_labels = classifier.classes_

        smooth_coefficient_count = (
            len(config.smooth_features)
            * n_splines_per_feature
        )

        smooth_coefficients = coefficients[
            :,
            :smooth_coefficient_count,
        ]

        lower_q, upper_q = central_quantile_range

        for feature_index, feature_name in enumerate(
            config.smooth_features
        ):
            observed_values = (
                self.X[feature_name]
                .astype(float)
                .to_numpy()
            )

            lower = float(
                np.quantile(observed_values, lower_q)
            )
            upper = float(
                np.quantile(observed_values, upper_q)
            )

            grid = np.linspace(lower, upper, grid_size)

            # SplineTransformer was fitted jointly to all smooth
            # columns. Build matrices where only the feature of
            # interest varies.
            reference_values = np.column_stack(
                [
                    np.full(
                        grid_size,
                        float(
                            self.X[column]
                            .astype(float)
                            .median()
                        ),
                    )
                    for column in config.smooth_features
                ]
            )

            reference_values[:, feature_index] = grid

            transformed_grid = (
                fitted_spline_transformer.transform(
                    reference_values
                )
            )

            start = (
                feature_index * n_splines_per_feature
            )
            stop = start + n_splines_per_feature

            feature_basis = transformed_grid[:, start:stop]

            observed_reference = np.column_stack(
                [
                    np.full(
                        len(observed_values),
                        float(
                            self.X[column]
                            .astype(float)
                            .median()
                        ),
                    )
                    for column in config.smooth_features
                ]
            )

            observed_reference[:, feature_index] = (
                observed_values
            )

            transformed_observed = (
                fitted_spline_transformer.transform(
                    observed_reference
                )
            )

            observed_basis = transformed_observed[
                :,
                start:stop,
            ]

            figure, axis = plt.subplots(
                figsize=(9, 5.5)
            )

            for class_index, class_name in enumerate(
                class_labels
            ):
                beta = smooth_coefficients[
                    class_index,
                    start:stop,
                ]

                contribution = feature_basis @ beta
                observed_contribution = (
                    observed_basis @ beta
                )

                contribution = (
                    contribution
                    - observed_contribution.mean()
                )

                axis.plot(
                    grid,
                    contribution,
                    linewidth=2,
                    label=str(class_name),
                )

            axis.axhline(
                0.0,
                color="black",
                linewidth=0.8,
                linestyle="--",
            )

            axis.set_title(
                f"Class-specific main effect of {feature_name}"
            )
            axis.set_xlabel(feature_name)
            axis.set_ylabel(
                "Centered additive score contribution"
            )
            axis.legend(title="Class")
            figure.tight_layout()

            figure.savefig(
                config.output_directory
                / f"main_effect_{feature_name}.png",
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(figure)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_labels: np.ndarray,
    ) -> None:
        figure, axis = plt.subplots(figsize=(6.5, 5.5))

        image = axis.imshow(matrix)

        axis.set_xticks(np.arange(len(class_labels)))
        axis.set_yticks(np.arange(len(class_labels)))
        axis.set_xticklabels(class_labels)
        axis.set_yticklabels(class_labels)

        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        axis.set_title(
            "Full-data confusion matrix\n"
            "(descriptive, not a generalization estimate)"
        )

        threshold = matrix.max() / 2

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column] > threshold
                        else "black"
                    ),
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()

        figure.savefig(
            self.config.output_directory
            / "full_data_confusion_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    def _require_loaded_data(self) -> None:
        if self.data is None or self.X is None or self.y is None:
            raise RuntimeError(
                "Call load_data() before this operation."
            )

    def _require_fitted_model(self) -> None:
        self._require_loaded_data()

        if self.best_model is None:
            raise RuntimeError(
                "Call fit_final_model() before this operation."
            )


# ============================================================
# Main execution
# ============================================================


def main() -> None:
    config = GAMConfig()

    analysis = ClassicalMultinomialGAM(config)

    analysis.load_data()

    audit = analysis.audit_data()

    print("\nData audit")
    print(json.dumps(audit, indent=2))

    print("\nRunning repeated nested cross-validation...")
    cv_results = analysis.evaluate_nested_cv()

    metric_columns = [
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    print("\nNested cross-validation summary")
    print(
        cv_results[metric_columns]
        .agg(["mean", "std"])
        .T
        .round(4)
    )

    print("\nFitting the final descriptive model...")
    search = analysis.fit_final_model()

    print("\nBest hyperparameters")
    print(search.best_params_)

    descriptive_metrics = (
        analysis.full_data_diagnostics()
    )

    print("\nFull-data descriptive metrics")
    print(
        json.dumps(
            descriptive_metrics,
            indent=2,
        )
    )

    analysis.plot_main_effects()

    print(
        "\nAnalysis complete. Results were saved to:"
        f"\n{config.output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()