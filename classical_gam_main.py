from __future__ import annotations

# ============================================================
# Configure a non-interactive Matplotlib backend before pyplot.
# ============================================================

import os

os.environ["MPLBACKEND"] = "Agg"

import json
import platform
import sys
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
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
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)


PREDICTOR_COLUMNS: tuple[str, ...] = (
    "X1", "X2", "X3", "X4", "X5", "X6", "X7",
)
RESPONSE_COLUMN = "Y"
EXPECTED_COLUMNS: tuple[str, ...] = (*PREDICTOR_COLUMNS, RESPONSE_COLUMN)


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class GAMConfig:
    """Configuration for the classical main-effects GAM analysis."""

    data_path: Path = Path("data/dataset.csv")
    output_directory: Path = Path("outputs")
    target_column: str = RESPONSE_COLUMN

    # Current requested model representation.
    smooth_features: tuple[str, ...] = ("X1", "X2", "X4", "X5", "X7",)
    linear_features: tuple[str, ...] = ("X6",)
    categorical_features: tuple[str, ...] = ("X3",)

    class_order: tuple[str, ...] = ("O", "M",)
    random_state: int = 42

    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 5

    n_knots_grid: tuple[int, ...] = (4, 5, 6, 7, 8,)
    degree_grid: tuple[int, ...] = (2, 3,)
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0,)

    # Keep only one CV level parallelized.
    inner_n_jobs: int = 1
    outer_n_jobs: int = 1


# ============================================================
# Classical main-effects GAM
# ============================================================

class ClassicalMultinomialGAM:
    """Penalized logistic additive B-spline model with main effects."""

    def __init__(self, config: GAMConfig) -> None:
        self.config = config
        self.data: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.grid_search: GridSearchCV | None = None
        self.best_model: Pipeline | None = None

    # --------------------------------------------------------
    # Data loading and validation
    # --------------------------------------------------------

    def load_data(self) -> None:
        """Load the CSV, validate it, and construct X and y."""
        config = self.config
        data_path = Path(config.data_path)

        print("\nLoading data from:")
        print(data_path.resolve())

        if not data_path.exists():
            raise FileNotFoundError(
                "Dataset file was not found:\n"
                f"{data_path.resolve()}"
            )

        data = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
            skipinitialspace=True,
        )

        if data.empty:
            raise ValueError("The dataset contains no observations.")

        data.columns = (
            data.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        expected_columns = list(EXPECTED_COLUMNS)
        observed_columns = data.columns.tolist()
        missing_columns = [
            column for column in expected_columns
            if column not in observed_columns
        ]
        unexpected_columns = [
            column for column in observed_columns
            if column not in expected_columns
        ]

        if missing_columns:
            raise ValueError(
                "The dataset is missing required columns.\n"
                f"Missing: {missing_columns}\n"
                f"Observed: {observed_columns}"
            )
        if unexpected_columns:
            raise ValueError(
                "The dataset contains unexpected columns.\n"
                f"Unexpected: {unexpected_columns}\n"
                f"Expected: {expected_columns}"
            )

        data = data.loc[:, expected_columns].copy()
        predictor_columns = list(PREDICTOR_COLUMNS)
        original_predictors = data.loc[:, predictor_columns].copy()

        for column in predictor_columns:
            if (
                pd.api.types.is_object_dtype(data[column].dtype)
                or pd.api.types.is_string_dtype(data[column].dtype)
            ):
                data[column] = data[column].astype("string").str.strip()
            data[column] = pd.to_numeric(data[column], errors="coerce")

        invalid_mask = (
            data[predictor_columns].isna()
            & original_predictors.notna()
        )
        if invalid_mask.any().any():
            details = self._format_bad_cells(
                invalid_mask, original_predictors, predictor_columns
            )
            raise ValueError(
                "Non-numeric predictor values were found:\n" + details
            )

        missing_mask = data[predictor_columns].isna()
        if missing_mask.any().any():
            details = self._format_bad_cells(
                missing_mask, data[predictor_columns], predictor_columns
            )
            raise ValueError(
                "Missing predictor values were found:\n" + details
            )

        data[config.target_column] = (
            data[config.target_column].astype("string").str.strip()
        )
        missing_response = (
            data[config.target_column].isna()
            | data[config.target_column].eq("")
        )
        if missing_response.any():
            csv_rows = (np.flatnonzero(missing_response.to_numpy()) + 2).tolist()
            raise ValueError(
                "Missing response labels were found in CSV rows: "
                f"{csv_rows[:20]}"
            )

        non_numeric = [
            column for column in predictor_columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if non_numeric:
            dtypes = {column: str(data[column].dtype) for column in non_numeric}
            raise TypeError(
                "Predictors remained non-numeric after conversion: "
                f"{dtypes}"
            )

        predictor_array = data[predictor_columns].to_numpy(dtype=float)
        if not np.isfinite(predictor_array).all():
            locations = np.argwhere(~np.isfinite(predictor_array))
            details = []
            for row_position, column_position in locations[:20]:
                details.append(
                    f"CSV row {int(row_position) + 2}, "
                    f"column {predictor_columns[int(column_position)]}: "
                    f"{predictor_array[row_position, column_position]!r}"
                )
            raise ValueError(
                "Non-finite predictor values were found:\n"
                + "\n".join(details)
            )

        observed_classes = set(
            data[config.target_column].dropna().unique().tolist()
        )
        expected_classes = set(config.class_order)
        unexpected_classes = observed_classes - expected_classes
        missing_classes = expected_classes - observed_classes
        if unexpected_classes:
            raise ValueError(
                "Unexpected response classes: "
                f"{sorted(unexpected_classes)}; expected "
                f"{list(config.class_order)}"
            )
        if missing_classes:
            raise ValueError(
                "Configured response classes are absent: "
                f"{sorted(missing_classes)}; observed "
                f"{sorted(observed_classes)}"
            )

        feature_groups = {
            "smooth_features": config.smooth_features,
            "linear_features": config.linear_features,
            "categorical_features": config.categorical_features,
        }

        for group_name, features in feature_groups.items():
            if isinstance(features, str):
                raise TypeError(
                    f"GAMConfig.{group_name} must be a tuple of "
                    "feature names, but it is a string:\n"
                    f"{features!r}\n"
                    "For a one-element tuple, include a trailing "
                    "comma, for example ('X6',)."
                )

            if not isinstance(features, tuple):
                raise TypeError(
                    f"GAMConfig.{group_name} must be a tuple, "
                    f"not {type(features).__name__}."
                )

            invalid_names = [
                feature
                for feature in features
                if not isinstance(feature, str)
            ]

            if invalid_names:
                raise TypeError(
                    f"GAMConfig.{group_name} contains non-string "
                    f"feature names: {invalid_names}"
                )

        configured_features = (
            config.smooth_features
            + config.linear_features
            + config.categorical_features
        )

        duplicate_features = sorted(
            {
                feature
                for feature in configured_features
                if configured_features.count(feature) > 1
            }
        )

        if duplicate_features:
            raise ValueError(
                "Predictors occur in more than one feature group: "
                f"{duplicate_features}"
            )

        unknown_features = [
            feature
            for feature in configured_features
            if feature not in predictor_columns
        ]

        if unknown_features:
            raise ValueError(
                "Configured model features are not available "
                "predictors: "
                f"{unknown_features}"
            )

        unused_features = [
            feature
            for feature in predictor_columns
            if feature not in configured_features
        ]

        if unused_features:
            raise ValueError(
                "Predictors are not assigned to a model feature "
                f"group: {unused_features}"
            )

        duplicate_features = sorted({
            feature for feature in configured_features
            if configured_features.count(feature) > 1
        })
        if duplicate_features:
            raise ValueError(
                "Predictors occur in more than one feature group: "
                f"{duplicate_features}"
            )
        unknown_features = [
            feature for feature in configured_features
            if feature not in predictor_columns
        ]
        if unknown_features:
            raise ValueError(
                "Configured model features are not predictors: "
                f"{unknown_features}"
            )
        unused_features = [
            feature for feature in predictor_columns
            if feature not in configured_features
        ]
        if unused_features:
            raise ValueError(
                "Predictors are not assigned to a feature group: "
                f"{unused_features}"
            )

        config.output_directory.mkdir(parents=True, exist_ok=True)
        self.data = data

        model_columns = list(configured_features)
        self.X = data.loc[:, model_columns].copy()
        self.y = data[config.target_column].copy()

        # Values are validated numerically first, then represented as
        # stable strings only in the model matrix for OneHotEncoder.
        for column in config.categorical_features:
            self.X[column] = self.X[column].map(
                lambda value: format(float(value), "g")
            )

        print("\nDataset loaded successfully.")
        print("Dataset shape:", self.data.shape)
        print("\nValidated source-data dtypes:")
        print(self.data.dtypes.to_string())
        print("\nModel-matrix dtypes:")
        print(self.X.dtypes.to_string())
        print("\nObserved response classes:")
        print(self.y.value_counts().sort_index().to_string())

    @staticmethod
    def _format_bad_cells(
        mask: pd.DataFrame,
        values: pd.DataFrame,
        columns: list[str],
        maximum: int = 20,
    ) -> str:
        positions = np.argwhere(mask.to_numpy())
        details: list[str] = []
        for row_position, column_position in positions[:maximum]:
            details.append(
                f"CSV row {int(row_position) + 2}, "
                f"column {columns[int(column_position)]}: "
                f"{values.iloc[row_position, column_position]!r}"
            )
        if len(positions) > maximum:
            details.append(f"... and {len(positions) - maximum} more cells")
        return "\n".join(details)

    def save_environment_information(self) -> dict[str, str]:
        """Save package, interpreter, platform, and backend information."""
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        information = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "operating_system": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "matplotlib_version": matplotlib.__version__,
            "joblib_version": joblib.__version__,
            "matplotlib_backend": str(matplotlib.get_backend()),
        }
        path = self.config.output_directory / "environment_information.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(information, file, indent=2, ensure_ascii=False)
        return information

    # --------------------------------------------------------
    # Data audit
    # --------------------------------------------------------

    def audit_data(self) -> dict[str, Any]:
        """Audit dtypes, classes, duplicates, distributions, and X5."""
        if self.data is None:
            raise RuntimeError("Call load_data() before audit_data().")

        data = self.data
        predictor_columns = list(PREDICTOR_COLUMNS)
        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(f"Loaded data are missing: {missing_columns}")

        non_numeric = [
            column for column in predictor_columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if non_numeric:
            dtypes = {column: str(data[column].dtype) for column in non_numeric}
            raise TypeError(
                "audit_data() requires numeric predictors: "
                f"{dtypes}"
            )

        row_count = int(len(data))
        if row_count == 0:
            raise ValueError("Cannot audit an empty dataset.")

        missing_counts = data.isna().sum().astype(int).to_dict()
        unique_counts = (
            data[predictor_columns].nunique(dropna=False).astype(int).to_dict()
        )
        class_counts_series = (
            data[RESPONSE_COLUMN].value_counts(dropna=False).sort_index()
        )
        class_counts = {
            str(name): int(count)
            for name, count in class_counts_series.items()
        }
        class_proportions = {
            str(name): float(count / row_count)
            for name, count in class_counts_series.items()
        }

        exact_duplicate_mask = data.duplicated(keep=False)
        exact_duplicate_rows = data.loc[exact_duplicate_mask].copy()
        duplicate_predictor_mask = data.duplicated(
            subset=predictor_columns, keep=False
        )
        duplicate_predictor_rows = data.loc[duplicate_predictor_mask].copy()
        duplicate_predictor_groups = (
            duplicate_predictor_rows[predictor_columns].drop_duplicates()
        )

        labels_per_configuration = (
            data.groupby(
                predictor_columns,
                dropna=False,
                observed=True,
            )[RESPONSE_COLUMN]
            .nunique(dropna=False)
        )
        conflicting_configurations = labels_per_configuration[
            labels_per_configuration > 1
        ]

        unusual_x5_mask = data["X5"].gt(40.0)
        unusual_x5_rows = data.loc[unusual_x5_mask].copy()
        descriptive_statistics = data[predictor_columns].describe().T
        correlation_matrix = data[predictor_columns].corr(method="pearson")

        audit: dict[str, Any] = {
            "row_count": row_count,
            "column_count": int(data.shape[1]),
            "columns": data.columns.tolist(),
            "dtypes": {
                column: str(dtype) for column, dtype in data.dtypes.items()
            },
            "missing_value_counts": missing_counts,
            "predictor_unique_counts": unique_counts,
            "class_counts": class_counts,
            "class_proportions": class_proportions,
            "exact_duplicate_row_count": int(exact_duplicate_mask.sum()),
            "duplicate_predictor_row_count": int(
                duplicate_predictor_mask.sum()
            ),
            "duplicate_predictor_group_count": int(
                len(duplicate_predictor_groups)
            ),
            "conflicting_predictor_configuration_count": int(
                len(conflicting_configurations)
            ),
            "unusual_x5_threshold": 40.0,
            "unusual_x5_count": int(unusual_x5_mask.sum()),
            "unusual_x5_indices": unusual_x5_rows.index.astype(int).tolist(),
        }

        print(f"Observations: {row_count}")
        print(f"Columns: {data.shape[1]}")
        print("\nMissing values:")
        print(pd.Series(missing_counts, name="missing_count").to_string())
        print("\nClass distribution:")
        print(pd.DataFrame({
            "count": pd.Series(class_counts),
            "proportion": pd.Series(class_proportions),
        }).to_string())
        print("\nUnique predictor values:")
        print(pd.Series(unique_counts, name="unique_count").to_string())
        print(
            "\nNumber of exact duplicate rows: "
            f"{int(exact_duplicate_mask.sum())}"
        )
        print(
            "Rows belonging to duplicate predictor configurations: "
            f"{int(duplicate_predictor_mask.sum())}"
        )
        print(
            "Duplicate predictor groups: "
            f"{len(duplicate_predictor_groups)}"
        )
        print(
            "Predictor configurations with conflicting labels: "
            f"{len(conflicting_configurations)}"
        )
        print(f"\nObservations with X5 > 40: {int(unusual_x5_mask.sum())}")
        if unusual_x5_mask.any():
            print("\nRows with X5 > 40:")
            print(unusual_x5_rows.to_string(index=True))

        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        with (output / "data_audit.json").open("w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2, ensure_ascii=False)
        descriptive_statistics.to_csv(output / "descriptive_statistics.csv")
        correlation_matrix.to_csv(output / "predictor_correlations.csv")
        unusual_x5_rows.to_csv(
            output / "unusual_x5_observations.csv",
            index=True,
            index_label="row_id",
        )
        exact_duplicate_rows.to_csv(
            output / "exact_duplicate_rows.csv",
            index=True,
            index_label="row_id",
        )
        duplicate_predictor_rows.to_csv(
            output / "duplicate_predictor_rows.csv",
            index=True,
            index_label="row_id",
        )
        if len(conflicting_configurations):
            conflicting_configurations.rename("number_of_labels").to_csv(
                output / "conflicting_predictor_configurations.csv"
            )

        self._plot_class_distribution(class_counts_series)
        self._plot_correlation_matrix(correlation_matrix)
        print("\nData audit completed successfully.")
        return audit

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------

    def build_pipeline(self) -> Pipeline:
        """
        Construct the main-effects-only additive model.
        """

        config = self.config

        spline_transformer = SplineTransformer(
            n_knots=5,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
            order="C",
        )

        linear_transformer = StandardScaler()

        categorical_transformer = OneHotEncoder(
            drop=None,
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
                    "linear",
                    linear_transformer,
                    list(config.linear_features),
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
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )


    def parameter_grid(self) -> dict[str, list[Any]]:
        config = self.config
        return {
            "preprocessor__smooth__n_knots": list(config.n_knots_grid),
            "preprocessor__smooth__degree": list(config.degree_grid),
            "classifier__C": list(config.c_grid),
        }

    def create_inner_cv(self, random_state: int | None = None) -> StratifiedKFold:
        if random_state is None:
            random_state = self.config.random_state
        return StratifiedKFold(
            n_splits=self.config.inner_splits,
            shuffle=True,
            random_state=random_state,
        )

    # --------------------------------------------------------
    # Repeated nested cross-validation
    # --------------------------------------------------------

    def evaluate_nested_cv(self) -> pd.DataFrame:
        """Estimate generalization with repeated nested CV."""
        self._require_loaded_data()
        assert self.X is not None
        assert self.y is not None
        config = self.config

        if self.y.value_counts().min() < config.outer_splits:
            raise ValueError(
                "The smallest class has fewer observations than "
                "outer_splits."
            )
        smallest_outer_training_class = int(
            np.floor(
                self.y.value_counts().min()
                * (config.outer_splits - 1)
                / config.outer_splits
            )
        )
        if smallest_outer_training_class < config.inner_splits:
            raise ValueError(
                "An outer-training class may have fewer observations "
                "than inner_splits."
            )

        inner_search = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=self.create_inner_cv(),
            refit=True,
            n_jobs=config.inner_n_jobs,
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
            inner_search,
            self.X,
            self.y,
            scoring=scoring,
            cv=outer_cv,
            n_jobs=config.outer_n_jobs,
            return_estimator=True,
            return_train_score=False,
            error_score="raise",
        )

        number_of_folds = len(results["test_accuracy"])
        zero_based = np.arange(number_of_folds)
        fold_results = pd.DataFrame({
            "fold": zero_based + 1,
            "repeat": zero_based // config.outer_splits + 1,
            "fold_within_repeat": zero_based % config.outer_splits + 1,
            "log_loss": -results["test_log_loss"],
            "accuracy": results["test_accuracy"],
            "balanced_accuracy": results["test_balanced_accuracy"],
            "macro_f1": results["test_macro_f1"],
            "fit_time_seconds": results["fit_time"],
            "score_time_seconds": results["score_time"],
        })
        searches = results["estimator"]
        parameters = [search.best_params_ for search in searches]
        fold_results["best_n_knots"] = [
            item["preprocessor__smooth__n_knots"] for item in parameters
        ]
        fold_results["best_degree"] = [
            item["preprocessor__smooth__degree"] for item in parameters
        ]
        fold_results["best_C"] = [
            item["classifier__C"] for item in parameters
        ]
        fold_results["best_inner_log_loss"] = [
            -float(search.best_score_) for search in searches
        ]

        output = config.output_directory
        fold_results.to_csv(output / "nested_cv_fold_results.csv", index=False)
        metrics = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
        summary = fold_results[metrics].agg(
            ["mean", "std", "min", "median", "max"]
        ).T
        summary["standard_error"] = summary["std"] / np.sqrt(number_of_folds)
        summary["ci_95_lower"] = summary["mean"] - 1.96 * summary["standard_error"]
        summary["ci_95_upper"] = summary["mean"] + 1.96 * summary["standard_error"]
        summary.to_csv(output / "nested_cv_summary.csv")
        self._create_selection_frequency_table(fold_results).to_csv(
            output / "nested_cv_hyperparameter_frequencies.csv",
            index=False,
        )
        self._plot_nested_cv_metrics(fold_results)
        self._plot_hyperparameter_frequencies(fold_results)
        return fold_results

    # --------------------------------------------------------
    # Final full-data model and diagnostics
    # --------------------------------------------------------

    def fit_final_model(self) -> GridSearchCV:
        self._require_loaded_data()
        assert self.X is not None
        assert self.y is not None
        config = self.config
        self.grid_search = GridSearchCV(
            self.build_pipeline(),
            self.parameter_grid(),
            scoring="neg_log_loss",
            cv=self.create_inner_cv(),
            refit=True,
            n_jobs=config.inner_n_jobs,
            return_train_score=True,
            error_score="raise",
        )
        self.grid_search.fit(self.X, self.y)
        self.best_model = self.grid_search.best_estimator_
        joblib.dump(
            self.best_model,
            config.output_directory / "classical_gam_main_effects.joblib",
        )
        best_parameters = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in self.grid_search.best_params_.items()
        }
        with (config.output_directory / "best_hyperparameters.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(best_parameters, file, indent=2)
        pd.DataFrame(self.grid_search.cv_results_).to_csv(
            config.output_directory / "final_model_grid_search_results.csv",
            index=False,
        )
        return self.grid_search

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
            raise RuntimeError("Predicted probabilities do not sum to one.")

        metrics = {
            "accuracy": float(accuracy_score(self.y, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(self.y, predictions)
            ),
            "macro_f1": float(f1_score(
                self.y,
                predictions,
                average="macro",
                zero_division=0,
            )),
            "multiclass_log_loss": float(log_loss(
                self.y,
                probabilities,
                labels=class_labels,
            )),
            "maximum_probability_sum_error": row_sum_error,
        }
        with (config.output_directory / "full_data_descriptive_metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metrics, file, indent=2)

        report = classification_report(
            self.y,
            predictions,
            labels=list(class_labels),
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(
            config.output_directory / "full_data_classification_report.csv"
        )
        matrix = confusion_matrix(self.y, predictions, labels=class_labels)
        pd.DataFrame(
            matrix,
            index=[f"Observed_{label}" for label in class_labels],
            columns=[f"Predicted_{label}" for label in class_labels],
        ).to_csv(config.output_directory / "full_data_confusion_matrix.csv")

        prediction_output = self.X.copy()
        prediction_output["observed_class"] = self.y.to_numpy()
        prediction_output["predicted_class"] = predictions
        for index, name in enumerate(class_labels):
            prediction_output[f"probability_{name}"] = probabilities[:, index]
        prediction_output["maximum_probability"] = probabilities.max(axis=1)
        prediction_output["correct"] = (
            prediction_output["observed_class"]
            == prediction_output["predicted_class"]
        )
        prediction_output.to_csv(
            config.output_directory / "full_data_fitted_predictions.csv",
            index=False,
        )
        self._plot_confusion_matrix(matrix, class_labels)
        self._export_transformed_coefficients()
        return metrics

    # --------------------------------------------------------
    # Main-effect plots and coefficient export
    # --------------------------------------------------------

    def plot_main_effects(
        self,
        grid_size: int = 200,
        central_quantile_range: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        self._require_fitted_model()
        assert self.best_model is not None
        assert self.X is not None
        config = self.config
        preprocessor = self.best_model.named_steps["preprocessor"]
        classifier = self.best_model.named_steps["classifier"]
        spline_transformer = preprocessor.named_transformers_["smooth"]
        class_labels, coefficients, _ = self._expanded_class_parameters(
            classifier
        )
        smooth_slice = preprocessor.output_indices_["smooth"]
        smooth_coefficients = coefficients[:, smooth_slice]
        number_of_features = len(config.smooth_features)
        if number_of_features == 0:
            return
        total_columns = smooth_coefficients.shape[1]
        if total_columns % number_of_features != 0:
            raise RuntimeError(
                "Spline output cannot be divided among smooth predictors."
            )
        splines_per_feature = total_columns // number_of_features
        lower_quantile, upper_quantile = central_quantile_range

        for feature_index, feature_name in enumerate(config.smooth_features):
            observed = self.X[feature_name].astype(float).to_numpy()
            grid = np.linspace(
                float(np.quantile(observed, lower_quantile)),
                float(np.quantile(observed, upper_quantile)),
                grid_size,
            )
            grid_reference = pd.DataFrame({
                column: np.full(
                    grid_size,
                    float(self.X[column].astype(float).median()),
                )
                for column in config.smooth_features
            })
            grid_reference.loc[:, feature_name] = grid
            transformed_grid = spline_transformer.transform(grid_reference)

            observed_reference = pd.DataFrame({
                column: np.full(
                    len(observed),
                    float(self.X[column].astype(float).median()),
                )
                for column in config.smooth_features
            })
            observed_reference.loc[:, feature_name] = observed
            transformed_observed = spline_transformer.transform(
                observed_reference
            )
            start = feature_index * splines_per_feature
            stop = start + splines_per_feature
            grid_basis = transformed_grid[:, start:stop]
            observed_basis = transformed_observed[:, start:stop]

            figure, axis = plt.subplots(figsize=(9, 5.5))
            for class_index, class_name in enumerate(class_labels):
                local = smooth_coefficients[class_index, start:stop]
                contribution = grid_basis @ local
                center = float((observed_basis @ local).mean())
                axis.plot(
                    grid,
                    contribution - center,
                    linewidth=2,
                    label=str(class_name),
                )
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"Class-specific main effect of {feature_name}")
            axis.set_xlabel(feature_name)
            axis.set_ylabel("Centered additive score contribution")
            axis.legend(title="Class")
            axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(
                config.output_directory / f"main_effect_{feature_name}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

    @staticmethod
    def _expanded_class_parameters(
        classifier: LogisticRegression,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one mathematically equivalent score row per class."""
        labels = np.asarray(classifier.classes_)
        coefficients = np.asarray(classifier.coef_, dtype=float)
        intercepts = np.asarray(classifier.intercept_, dtype=float)
        if len(labels) == 2:
            if coefficients.shape[0] != 1 or intercepts.shape[0] != 1:
                raise RuntimeError(
                    "Unexpected binary logistic parameter shapes: "
                    f"coef={coefficients.shape}, "
                    f"intercept={intercepts.shape}"
                )
            expanded_coefficients = np.vstack([
                -0.5 * coefficients[0],
                0.5 * coefficients[0],
            ])
            expanded_intercepts = np.array([
                -0.5 * intercepts[0],
                0.5 * intercepts[0],
            ])
            return labels, expanded_coefficients, expanded_intercepts
        if coefficients.shape[0] != len(labels):
            raise RuntimeError(
                "Coefficient rows do not match classifier classes."
            )
        return labels, coefficients, intercepts

    def _export_transformed_coefficients(self) -> None:
        self._require_fitted_model()
        assert self.best_model is not None
        preprocessor = self.best_model.named_steps["preprocessor"]
        classifier = self.best_model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        labels, coefficients, intercepts = self._expanded_class_parameters(
            classifier
        )
        frames = []
        for index, name in enumerate(labels):
            frames.append(pd.DataFrame({
                "class": str(name),
                "transformed_feature": feature_names,
                "coefficient": coefficients[index],
            }))
        table = pd.concat(frames, ignore_index=True)
        table["absolute_coefficient"] = table["coefficient"].abs()
        table = table.sort_values(
            ["class", "absolute_coefficient"],
            ascending=[True, False],
        ).reset_index(drop=True)
        table.to_csv(
            self.config.output_directory / "transformed_feature_coefficients.csv",
            index=False,
        )
        pd.DataFrame({
            "class": labels,
            "intercept": intercepts,
        }).to_csv(
            self.config.output_directory / "class_intercepts.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Tables and plots
    # --------------------------------------------------------

    @staticmethod
    def _create_selection_frequency_table(
        fold_results: pd.DataFrame,
    ) -> pd.DataFrame:
        frames = []
        for column in ["best_n_knots", "best_degree", "best_C"]:
            counts = fold_results[column].value_counts().sort_index()
            frames.append(pd.DataFrame({
                "hyperparameter": column,
                "value": counts.index.astype(str),
                "count": counts.to_numpy(),
                "proportion": counts.to_numpy() / counts.sum(),
            }))
        return pd.concat(frames, ignore_index=True)

    def _plot_class_distribution(self, class_counts: pd.Series) -> None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        positions = np.arange(len(class_counts))
        bars = axis.bar(positions, class_counts.to_numpy())
        axis.set_xticks(positions)
        axis.set_xticklabels(class_counts.index.astype(str))
        axis.set_xlabel("Class")
        axis.set_ylabel("Number of observations")
        axis.set_title("Class distribution")
        for bar, count in zip(bars, class_counts.to_numpy(), strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
            )
        figure.tight_layout()
        figure.savefig(
            self.config.output_directory / "class_distribution.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _plot_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1)
        labels = matrix.columns
        axis.set_xticks(np.arange(len(labels)))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticklabels(labels)
        axis.set_title("Predictor correlation matrix")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                )
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        figure.tight_layout()
        figure.savefig(
            self.config.output_directory / "correlation_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _plot_nested_cv_metrics(self, fold_results: pd.DataFrame) -> None:
        metrics = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for axis, metric in zip(axes.ravel(), metrics, strict=True):
            axis.plot(
                fold_results["fold"],
                fold_results[metric],
                marker="o",
                linewidth=1,
                markersize=4,
            )
            axis.axhline(
                fold_results[metric].mean(),
                linestyle="--",
                linewidth=1,
                color="black",
                label="Mean",
            )
            axis.set_title(metric)
            axis.set_xlabel("Outer fold")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Repeated nested cross-validation metrics")
        figure.tight_layout()
        figure.savefig(
            self.config.output_directory / "nested_cv_metrics.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _plot_hyperparameter_frequencies(
        self,
        fold_results: pd.DataFrame,
    ) -> None:
        settings = [
            ("best_n_knots", "Selected number of knots"),
            ("best_degree", "Selected spline degree"),
            ("best_C", "Selected regularization parameter C"),
        ]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for axis, (column, title) in zip(axes, settings, strict=True):
            counts = fold_results[column].value_counts().sort_index()
            positions = np.arange(len(counts))
            axis.bar(positions, counts.to_numpy())
            axis.set_xticks(positions)
            axis.set_xticklabels([str(value) for value in counts.index])
            axis.set_title(title)
            axis.set_xlabel("Value")
            axis.set_ylabel("Outer-fold selections")
        figure.tight_layout()
        figure.savefig(
            self.config.output_directory
            / "hyperparameter_selection_frequencies.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_labels: np.ndarray,
    ) -> None:
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix)
        positions = np.arange(len(class_labels))
        axis.set_xticks(positions)
        axis.set_yticks(positions)
        axis.set_xticklabels(class_labels)
        axis.set_yticklabels(class_labels)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        axis.set_title(
            "Full-data confusion matrix\n"
            "(descriptive, not a generalization estimate)"
        )
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white" if matrix[row, column] > threshold else "black"
                    ),
                )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(
            self.config.output_directory / "full_data_confusion_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    # --------------------------------------------------------
    # Validation helpers
    # --------------------------------------------------------

    def _require_loaded_data(self) -> None:
        if self.data is None or self.X is None or self.y is None:
            raise RuntimeError("Call load_data() before this operation.")

    def _require_fitted_model(self) -> None:
        self._require_loaded_data()
        if self.best_model is None:
            raise RuntimeError("Call fit_final_model() before this operation.")


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    config = GAMConfig()

    print("=" * 60)
    print("Classical logistic GAM with main effects")
    print("=" * 60)
    print("\nMatplotlib backend:", matplotlib.get_backend())
    print("Python executable:", sys.executable)

    analysis = ClassicalMultinomialGAM(config=config)
    analysis.load_data()

    environment_information = analysis.save_environment_information()
    print("\nExecution environment")
    for key, value in environment_information.items():
        print(f"{key}: {value}")

    print("\nRunning data audit...")
    audit = analysis.audit_data()
    print("\nData audit")
    print(json.dumps(audit, indent=2))

    print("\nRunning repeated nested cross-validation...")
    fold_results = analysis.evaluate_nested_cv()
    metric_columns = [
        "log_loss", "accuracy", "balanced_accuracy", "macro_f1",
    ]
    nested_summary = fold_results[metric_columns].agg(["mean", "std"]).T
    print("\nNested cross-validation summary")
    print(nested_summary.round(4))

    print("\nHyperparameter selection frequencies")
    for column in ["best_n_knots", "best_degree", "best_C"]:
        print(f"\n{column}")
        print(fold_results[column].value_counts().sort_index())

    print("\nFitting the final descriptive model...")
    search = analysis.fit_final_model()
    print("\nBest hyperparameters")
    print(search.best_params_)

    descriptive_metrics = analysis.full_data_diagnostics()
    print("\nFull-data descriptive metrics")
    print(json.dumps(descriptive_metrics, indent=2))

    print("\nGenerating main-effect plots...")
    analysis.plot_main_effects()

    print("\nAnalysis complete. Results were saved to:")
    print(config.output_directory.resolve())


if __name__ == "__main__":
    freeze_support()
    main()
