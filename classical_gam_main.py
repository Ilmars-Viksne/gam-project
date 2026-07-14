from __future__ import annotations

# ============================================================
# Configure a non-interactive Matplotlib backend.
#
# This must happen before importing matplotlib.pyplot.
# It prevents Tkinter/TkAgg errors during joblib and
# multiprocessing operations on Windows.
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


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class GAMConfig:
    """
    Configuration for the classical multinomial GAM analysis.
    """

    data_path: Path = Path("data/dataset.csv")
    output_directory: Path = Path("outputs")

    target_column: str = "Y"

    # Predictors represented by univariate spline functions.
    smooth_features: tuple[str, ...] = (
        "X1",
        "X2",
        "X4",
        "X5",
        "X7",
    )

    # Low-resolution numerical predictor treated as a linear
    # main effect rather than as a categorical variable.
    linear_features: tuple[str, ...] = (
        "X6",
    )

    # Ordered, low-resolution measurement treated as a
    # categorical main effect in the initial model.
    categorical_features: tuple[str, ...] = (
        "X3",
    )

    # Preferred order for summaries and target validation.
    class_order: tuple[str, ...] = (
        "O",
        "B",
        "M",
        "G",
    )

    random_state: int = 42

    # Outer cross-validation estimates model generalization.
    outer_splits: int = 5
    outer_repeats: int = 5

    # Inner cross-validation selects model complexity.
    inner_splits: int = 5

    # Expanded grid following the initial boundary result.
    n_knots_grid: tuple[int, ...] = (
        4,
        5,
        6,
        7,
        8,
    )

    degree_grid: tuple[int, ...] = (
        2,
        3,
    )

    c_grid: tuple[float, ...] = (
        0.01,
        0.1,
        1.0,
        10.0,
        30.0,
        100.0,
    )

    # Keep serial during the stable Windows run.
    #
    # After verifying that the script works reliably with the
    # Agg backend, this can be changed to -1 for the inner
    # GridSearchCV operations.
    inner_n_jobs: int = 1

    # Keep the outer loop serial to avoid nested parallelism.
    outer_n_jobs: int = 1


# ============================================================
# Classical multinomial GAM
# ============================================================


class ClassicalMultinomialGAM:
    """
    Penalized multinomial logistic additive B-spline model.

    Model structure
    ---------------

    The class-specific linear score is:

        eta_k(x)
        =
        intercept_k
        + sum_j f_jk(x_j)
        + sum_l beta_lk * x_l
        + categorical effects

    where:

    - f_jk is represented by a univariate B-spline basis;
    - numerical linear variables are standardized;
    - categorical variables are one-hot encoded;
    - multinomial logistic regression estimates class scores;
    - L2 regularization controls coefficient magnitude;
    - no pairwise interaction features are constructed.

    Consequently, this is a main-effects-only additive model.
    """

    def __init__(self, config: GAMConfig) -> None:
        self.config = config

        self.data: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None

        self.grid_search: GridSearchCV | None = None
        self.best_model: Pipeline | None = None

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------

    def load_data(self) -> None:
        """
        Load, validate, and prepare the CSV dataset.
        """

        config = self.config

        print(f"\nLoading data from:\n{config.data_path.resolve()}")

        if not config.data_path.exists():
            raise FileNotFoundError(
                "The dataset was not found at:\n"
                f"{config.data_path.resolve()}"
            )

        df = pd.read_csv(config.data_path)

        predictor_columns = [
            *config.smooth_features,
            *config.linear_features,
            *config.categorical_features,
        ]

        required_columns = [
            *predictor_columns,
            config.target_column,
        ]

        missing_columns = sorted(
            set(required_columns) - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                "The following required columns are missing:\n"
                f"{missing_columns}"
            )

        # Retain only the columns used by this model.
        df = df.loc[:, required_columns].copy()

        # All seven predictors must initially be parseable as
        # numerical measurements.
        for column in predictor_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        missing_counts = df[required_columns].isna().sum()

        if missing_counts.any():
            raise ValueError(
                "Missing values were found:\n"
                f"{missing_counts[missing_counts > 0]}"
            )

        observed_classes = set(
            df[config.target_column].astype(str).unique()
        )

        expected_classes = set(config.class_order)

        unexpected_classes = (
            observed_classes - expected_classes
        )

        missing_classes = (
            expected_classes - observed_classes
        )

        if unexpected_classes:
            raise ValueError(
                "Unexpected target classes were found:\n"
                f"{sorted(unexpected_classes)}"
            )

        if missing_classes:
            raise ValueError(
                "The following expected classes are absent:\n"
                f"{sorted(missing_classes)}"
            )

        # Preserve a meaningful class order for data summaries.
        df[config.target_column] = pd.Categorical(
            df[config.target_column],
            categories=config.class_order,
            ordered=False,
        )

        # Only the categorical predictors are converted to
        # strings. X6 remains numerical and is standardized.
        for column in config.categorical_features:
            df[column] = df[column].astype(str)

        self.data = df

        self.X = df.drop(
            columns=config.target_column
        ).copy()

        # Scikit-learn's LogisticRegression will determine the
        # internal class ordering. The fitted classifier's
        # classes_ attribute is used consistently afterward.
        self.y = df[config.target_column].astype(str)

        config.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Environment information
    # --------------------------------------------------------

    def save_environment_information(self) -> dict[str, str]:
        """
        Save package and execution-environment information.
        """

        config = self.config

        information = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "operating_system": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "matplotlib_version": matplotlib.__version__,
            "joblib_version": joblib.__version__,
            "matplotlib_backend": matplotlib.get_backend(),
        }

        output_path = (
            config.output_directory
            / "environment_information.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                information,
                file,
                indent=2,
            )

        return information

    # --------------------------------------------------------
    # Data audit
    # --------------------------------------------------------

    def audit_data(self) -> dict[str, Any]:
        """
        Audit class counts, uniqueness, duplicates, outliers,
        and predictor correlations.
        """

        self._require_loaded_data()

        assert self.data is not None
        assert self.X is not None
        assert self.y is not None

        config = self.config

        predictor_columns = list(self.X.columns)

        numerical_predictors = [
            *config.smooth_features,
            *config.linear_features,
        ]

        class_counts = (
            self.y
            .value_counts()
            .reindex(
                config.class_order,
                fill_value=0,
            )
        )

        class_proportions = (
            class_counts / class_counts.sum()
        )

        unique_counts = self.X.nunique()

        # True for every row that belongs to an exact duplicate
        # predictor configuration.
        duplicated_row_mask = self.X.duplicated(
            keep=False
        )

        duplicate_groups = (
            self.data
            .groupby(
                predictor_columns,
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

        duplicate_groups = duplicate_groups.loc[
            duplicate_groups["number_of_rows"] > 1
        ].copy()

        conflicting_duplicates = (
            duplicate_groups.loc[
                duplicate_groups["number_of_labels"] > 1
            ].copy()
        )

        x5_unusual = self.data.loc[
            self.data["X5"] > 40
        ].copy()

        correlation_matrix = (
            self.data[numerical_predictors]
            .astype(float)
            .corr()
        )

        descriptive_statistics = (
            self.data[numerical_predictors]
            .astype(float)
            .describe()
            .T
        )

        audit = {
            "number_of_rows": int(
                len(self.data)
            ),
            "number_of_predictors": int(
                len(predictor_columns)
            ),
            "class_counts": {
                str(key): int(value)
                for key, value
                in class_counts.items()
            },
            "class_proportions": {
                str(key): float(value)
                for key, value
                in class_proportions.items()
            },
            "unique_values": {
                str(key): int(value)
                for key, value
                in unique_counts.items()
            },
            "rows_in_duplicate_configurations": int(
                duplicated_row_mask.sum()
            ),
            "number_of_duplicate_groups": int(
                len(duplicate_groups)
            ),
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
            config.output_directory
            / "class_distribution.csv",
            index=False,
        )

        unique_counts.rename(
            "number_of_unique_values"
        ).to_csv(
            config.output_directory
            / "unique_value_counts.csv",
            header=True,
        )

        correlation_matrix.to_csv(
            config.output_directory
            / "correlation_matrix.csv"
        )

        descriptive_statistics.to_csv(
            config.output_directory
            / "descriptive_statistics.csv"
        )

        duplicate_groups.to_csv(
            config.output_directory
            / "duplicate_groups.csv",
            index=False,
        )

        conflicting_duplicates.to_csv(
            config.output_directory
            / "conflicting_duplicate_groups.csv",
            index=False,
        )

        x5_unusual.to_csv(
            config.output_directory
            / "unusual_x5_observations.csv",
            index=False,
        )

        with (
            config.output_directory
            / "data_audit.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                audit,
                file,
                indent=2,
            )

        self._plot_class_distribution(
            class_counts=class_counts
        )

        self._plot_correlation_matrix(
            correlation_matrix=correlation_matrix
        )

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

        # Standardization makes the single X6 linear coefficient
        # numerically comparable and appropriately regularized.
        linear_transformer = StandardScaler()

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
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

    def parameter_grid(self) -> dict[str, list[Any]]:
        """
        Hyperparameter grid used in the inner CV loop.
        """

        config = self.config

        return {
            "preprocessor__smooth__n_knots": list(
                config.n_knots_grid
            ),
            "preprocessor__smooth__degree": list(
                config.degree_grid
            ),
            "classifier__C": list(
                config.c_grid
            ),
        }

    def create_inner_cv(
        self,
        random_state: int | None = None,
    ) -> StratifiedKFold:
        """
        Construct the inner stratified CV splitter.
        """

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
        """
        Estimate generalization performance using repeated,
        stratified nested cross-validation.

        Outer loop:
            estimates performance.

        Inner loop:
            selects spline degree, knot count, and C.
        """

        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config

        base_pipeline = self.build_pipeline()

        inner_cv = self.create_inner_cv()

        inner_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=inner_cv,
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
            estimator=inner_search,
            X=self.X,
            y=self.y,
            scoring=scoring,
            cv=outer_cv,
            n_jobs=config.outer_n_jobs,
            return_estimator=True,
            return_train_score=False,
            error_score="raise",
        )

        number_of_outer_folds = len(
            results["test_accuracy"]
        )

        fold_results = pd.DataFrame(
            {
                "fold": np.arange(
                    1,
                    number_of_outer_folds + 1,
                ),
                "repeat": (
                    np.arange(number_of_outer_folds)
                    // config.outer_splits
                    + 1
                ),
                "fold_within_repeat": (
                    np.arange(number_of_outer_folds)
                    % config.outer_splits
                    + 1
                ),
                "log_loss": -results[
                    "test_log_loss"
                ],
                "accuracy": results[
                    "test_accuracy"
                ],
                "balanced_accuracy": results[
                    "test_balanced_accuracy"
                ],
                "macro_f1": results[
                    "test_macro_f1"
                ],
                "fit_time_seconds": results[
                    "fit_time"
                ],
                "score_time_seconds": results[
                    "score_time"
                ],
            }
        )

        fitted_searches = results["estimator"]

        best_parameters = [
            fitted_search.best_params_
            for fitted_search in fitted_searches
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

        fold_results["best_inner_log_loss"] = [
            -float(fitted_search.best_score_)
            for fitted_search in fitted_searches
        ]

        fold_results.to_csv(
            config.output_directory
            / "nested_cv_fold_results.csv",
            index=False,
        )

        metric_columns = [
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]

        summary = (
            fold_results[metric_columns]
            .agg(
                [
                    "mean",
                    "std",
                    "min",
                    "median",
                    "max",
                ]
            )
            .T
        )

        summary["standard_error"] = (
            summary["std"]
            / np.sqrt(number_of_outer_folds)
        )

        # Normal-approximation interval for descriptive use.
        summary["ci_95_lower"] = (
            summary["mean"]
            - 1.96 * summary["standard_error"]
        )

        summary["ci_95_upper"] = (
            summary["mean"]
            + 1.96 * summary["standard_error"]
        )

        summary.to_csv(
            config.output_directory
            / "nested_cv_summary.csv"
        )

        selection_frequencies = self._create_selection_frequency_table(
            fold_results=fold_results
        )

        selection_frequencies.to_csv(
            config.output_directory
            / "nested_cv_hyperparameter_frequencies.csv",
            index=False,
        )

        self._plot_nested_cv_metrics(
            fold_results=fold_results
        )

        self._plot_hyperparameter_frequencies(
            fold_results=fold_results
        )

        return fold_results

    # --------------------------------------------------------
    # Final full-data model
    # --------------------------------------------------------

    def fit_final_model(self) -> GridSearchCV:
        """
        Tune and fit the final descriptive model using all rows.

        This model is appropriate for effect visualization and
        future prediction after the nested-CV evaluation has
        already been completed.
        """

        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config

        inner_cv = self.create_inner_cv()

        self.grid_search = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=inner_cv,
            refit=True,
            n_jobs=config.inner_n_jobs,
            return_train_score=True,
            error_score="raise",
        )

        self.grid_search.fit(
            self.X,
            self.y,
        )

        self.best_model = (
            self.grid_search.best_estimator_
        )

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
            for key, value
            in self.grid_search.best_params_.items()
        }

        with (
            config.output_directory
            / "best_hyperparameters.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                best_parameters,
                file,
                indent=2,
            )

        cv_results = pd.DataFrame(
            self.grid_search.cv_results_
        )

        cv_results.to_csv(
            config.output_directory
            / "final_model_grid_search_results.csv",
            index=False,
        )

        return self.grid_search

    # --------------------------------------------------------
    # Full-data descriptive diagnostics
    # --------------------------------------------------------

    def full_data_diagnostics(self) -> dict[str, Any]:
        """
        Calculate descriptive fitted-data metrics.

        These values are not estimates of generalization
        performance because the same observations were used to
        fit and evaluate the final model.
        """

        self._require_fitted_model()

        assert self.best_model is not None
        assert self.X is not None
        assert self.y is not None

        config = self.config

        probabilities = (
            self.best_model.predict_proba(self.X)
        )

        predictions = (
            self.best_model.predict(self.X)
        )

        classifier = self.best_model.named_steps[
            "classifier"
        ]

        class_labels = classifier.classes_

        probability_row_sums = probabilities.sum(
            axis=1
        )

        row_sum_error = float(
            np.abs(
                probability_row_sums - 1.0
            ).max()
        )

        if row_sum_error > 1e-10:
            raise RuntimeError(
                "Predicted class probabilities do not "
                "sum to one within numerical tolerance."
            )

        metrics = {
            "accuracy": float(
                accuracy_score(
                    self.y,
                    predictions,
                )
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
                    zero_division=0,
                )
            ),
            "multiclass_log_loss": float(
                log_loss(
                    self.y,
                    probabilities,
                    labels=class_labels,
                )
            ),
            "maximum_probability_sum_error": (
                row_sum_error
            ),
        }

        with (
            config.output_directory
            / "full_data_descriptive_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
            )

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

        matrix_frame = pd.DataFrame(
            matrix,
            index=[
                f"Observed_{label}"
                for label in class_labels
            ],
            columns=[
                f"Predicted_{label}"
                for label in class_labels
            ],
        )

        matrix_frame.to_csv(
            config.output_directory
            / "full_data_confusion_matrix.csv"
        )

        prediction_output = self.X.copy()

        prediction_output[
            "observed_class"
        ] = self.y.values

        prediction_output[
            "predicted_class"
        ] = predictions

        for class_index, class_name in enumerate(
            class_labels
        ):
            prediction_output[
                f"probability_{class_name}"
            ] = probabilities[:, class_index]

        prediction_output["maximum_probability"] = (
            probabilities.max(axis=1)
        )

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

        self._export_transformed_coefficients()

        return metrics

    # --------------------------------------------------------
    # Main-effect curves
    # --------------------------------------------------------

    def plot_main_effects(
        self,
        grid_size: int = 200,
        central_quantile_range: tuple[
            float,
            float,
        ] = (
            0.01,
            0.99,
        ),
    ) -> None:
        """
        Plot class-specific spline contributions.

        Each curve represents the contribution of one predictor
        to a class-specific multinomial score.

        The plotted contribution is centered using the empirical
        average contribution over the observed values.

        These curves are additive score contributions. They are
        not direct probability changes.
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

        spline_transformer = (
            preprocessor.named_transformers_[
                "smooth"
            ]
        )

        class_labels = classifier.classes_
        coefficients = classifier.coef_

        smooth_output_slice = (
            preprocessor.output_indices_["smooth"]
        )

        smooth_coefficients = coefficients[
            :,
            smooth_output_slice,
        ]

        number_of_smooth_features = len(
            config.smooth_features
        )

        total_spline_columns = (
            smooth_coefficients.shape[1]
        )

        if (
            total_spline_columns
            % number_of_smooth_features
            != 0
        ):
            raise RuntimeError(
                "Unable to divide the spline output columns "
                "equally among the smooth predictors."
            )

        splines_per_feature = (
            total_spline_columns
            // number_of_smooth_features
        )

        lower_quantile, upper_quantile = (
            central_quantile_range
        )

        for feature_index, feature_name in enumerate(
            config.smooth_features
        ):
            observed_values = (
                self.X[feature_name]
                .astype(float)
                .to_numpy()
            )

            lower_limit = float(
                np.quantile(
                    observed_values,
                    lower_quantile,
                )
            )

            upper_limit = float(
                np.quantile(
                    observed_values,
                    upper_quantile,
                )
            )

            grid = np.linspace(
                lower_limit,
                upper_limit,
                grid_size,
            )

            # Use a DataFrame with the exact feature names used
            # to fit SplineTransformer. This avoids the warning:
            #
            # "X does not have valid feature names..."
            grid_reference = pd.DataFrame(
                {
                    column: np.full(
                        grid_size,
                        float(
                            self.X[column]
                            .astype(float)
                            .median()
                        ),
                    )
                    for column
                    in config.smooth_features
                }
            )

            grid_reference.loc[
                :,
                feature_name,
            ] = grid

            transformed_grid = (
                spline_transformer.transform(
                    grid_reference
                )
            )

            observed_reference = pd.DataFrame(
                {
                    column: np.full(
                        len(observed_values),
                        float(
                            self.X[column]
                            .astype(float)
                            .median()
                        ),
                    )
                    for column
                    in config.smooth_features
                }
            )

            observed_reference.loc[
                :,
                feature_name,
            ] = observed_values

            transformed_observed = (
                spline_transformer.transform(
                    observed_reference
                )
            )

            local_start = (
                feature_index
                * splines_per_feature
            )

            local_stop = (
                local_start
                + splines_per_feature
            )

            grid_basis = transformed_grid[
                :,
                local_start:local_stop,
            ]

            observed_basis = transformed_observed[
                :,
                local_start:local_stop,
            ]

            figure, axis = plt.subplots(
                figsize=(9, 5.5)
            )

            for class_index, class_name in enumerate(
                class_labels
            ):
                local_coefficients = (
                    smooth_coefficients[
                        class_index,
                        local_start:local_stop,
                    ]
                )

                grid_contribution = (
                    grid_basis
                    @ local_coefficients
                )

                observed_contribution = (
                    observed_basis
                    @ local_coefficients
                )

                centered_contribution = (
                    grid_contribution
                    - observed_contribution.mean()
                )

                axis.plot(
                    grid,
                    centered_contribution,
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
                "Class-specific main effect of "
                f"{feature_name}"
            )

            axis.set_xlabel(feature_name)

            axis.set_ylabel(
                "Centered additive score contribution"
            )

            axis.legend(
                title="Class"
            )

            axis.grid(
                alpha=0.2
            )

            figure.tight_layout()

            figure.savefig(
                config.output_directory
                / f"main_effect_{feature_name}.png",
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(figure)

    # --------------------------------------------------------
    # Coefficient export
    # --------------------------------------------------------

    def _export_transformed_coefficients(self) -> None:
        """
        Export coefficients for all transformed features.

        The spline coefficients should not generally be
        interpreted one at a time. Their combined contribution
        is represented by the main-effect plots.
        """

        self._require_fitted_model()

        assert self.best_model is not None

        config = self.config

        preprocessor = self.best_model.named_steps[
            "preprocessor"
        ]

        classifier = self.best_model.named_steps[
            "classifier"
        ]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        coefficient_frames: list[pd.DataFrame] = []

        for class_index, class_name in enumerate(
            classifier.classes_
        ):
            class_frame = pd.DataFrame(
                {
                    "class": class_name,
                    "transformed_feature": feature_names,
                    "coefficient": (
                        classifier.coef_[
                            class_index,
                            :
                        ]
                    ),
                }
            )

            coefficient_frames.append(
                class_frame
            )

        coefficient_table = pd.concat(
            coefficient_frames,
            ignore_index=True,
        )

        coefficient_table[
            "absolute_coefficient"
        ] = coefficient_table[
            "coefficient"
        ].abs()

        coefficient_table = (
            coefficient_table
            .sort_values(
                by=[
                    "class",
                    "absolute_coefficient",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        coefficient_table.to_csv(
            config.output_directory
            / "transformed_feature_coefficients.csv",
            index=False,
        )

        intercept_table = pd.DataFrame(
            {
                "class": classifier.classes_,
                "intercept": classifier.intercept_,
            }
        )

        intercept_table.to_csv(
            config.output_directory
            / "class_intercepts.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Tables and plots
    # --------------------------------------------------------

    @staticmethod
    def _create_selection_frequency_table(
        fold_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a tidy hyperparameter selection-frequency table.
        """

        frames: list[pd.DataFrame] = []

        for column in [
            "best_n_knots",
            "best_degree",
            "best_C",
        ]:
            counts = (
                fold_results[column]
                .value_counts()
                .sort_index()
            )

            frame = pd.DataFrame(
                {
                    "hyperparameter": column,
                    "value": counts.index.astype(str),
                    "count": counts.values,
                    "proportion": (
                        counts.values
                        / counts.values.sum()
                    ),
                }
            )

            frames.append(frame)

        return pd.concat(
            frames,
            ignore_index=True,
        )

    def _plot_class_distribution(
        self,
        class_counts: pd.Series,
    ) -> None:
        figure, axis = plt.subplots(
            figsize=(7, 4.5)
        )

        positions = np.arange(
            len(class_counts)
        )

        bars = axis.bar(
            positions,
            class_counts.values,
        )

        axis.set_xticks(
            positions
        )

        axis.set_xticklabels(
            class_counts.index
        )

        axis.set_xlabel(
            "Class"
        )

        axis.set_ylabel(
            "Number of observations"
        )

        axis.set_title(
            "Class distribution"
        )

        for bar, count in zip(
            bars,
            class_counts.values,
            strict=True,
        ):
            axis.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
            )

        figure.tight_layout()

        figure.savefig(
            self.config.output_directory
            / "class_distribution.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    def _plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> None:
        figure, axis = plt.subplots(
            figsize=(7, 6)
        )

        image = axis.imshow(
            correlation_matrix.to_numpy(),
            vmin=-1,
            vmax=1,
        )

        labels = correlation_matrix.columns

        axis.set_xticks(
            np.arange(len(labels))
        )

        axis.set_yticks(
            np.arange(len(labels))
        )

        axis.set_xticklabels(
            labels,
            rotation=45,
            ha="right",
        )

        axis.set_yticklabels(
            labels
        )

        axis.set_title(
            "Predictor correlation matrix"
        )

        for row in range(
            correlation_matrix.shape[0]
        ):
            for column in range(
                correlation_matrix.shape[1]
            ):
                value = correlation_matrix.iloc[
                    row,
                    column,
                ]

                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

        figure.colorbar(
            image,
            ax=axis,
            label="Pearson correlation",
        )

        figure.tight_layout()

        figure.savefig(
            self.config.output_directory
            / "correlation_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    def _plot_nested_cv_metrics(
        self,
        fold_results: pd.DataFrame,
    ) -> None:
        metric_columns = [
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]

        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(11, 8),
        )

        for axis, metric in zip(
            axes.ravel(),
            metric_columns,
            strict=True,
        ):
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

        figure.suptitle(
            "Repeated nested cross-validation metrics"
        )

        figure.tight_layout()

        figure.savefig(
            self.config.output_directory
            / "nested_cv_metrics.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    def _plot_hyperparameter_frequencies(
        self,
        fold_results: pd.DataFrame,
    ) -> None:
        settings = [
            (
                "best_n_knots",
                "Selected number of knots",
            ),
            (
                "best_degree",
                "Selected spline degree",
            ),
            (
                "best_C",
                "Selected regularization parameter C",
            ),
        ]

        figure, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(14, 4.5),
        )

        for axis, (
            column,
            title,
        ) in zip(
            axes,
            settings,
            strict=True,
        ):
            counts = (
                fold_results[column]
                .value_counts()
                .sort_index()
            )

            positions = np.arange(
                len(counts)
            )

            axis.bar(
                positions,
                counts.values,
            )

            axis.set_xticks(
                positions
            )

            axis.set_xticklabels(
                [
                    str(value)
                    for value in counts.index
                ]
            )

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
        figure, axis = plt.subplots(
            figsize=(6.5, 5.5)
        )

        image = axis.imshow(
            matrix
        )

        positions = np.arange(
            len(class_labels)
        )

        axis.set_xticks(
            positions
        )

        axis.set_yticks(
            positions
        )

        axis.set_xticklabels(
            class_labels
        )

        axis.set_yticklabels(
            class_labels
        )

        axis.set_xlabel(
            "Predicted class"
        )

        axis.set_ylabel(
            "Observed class"
        )

        axis.set_title(
            "Full-data confusion matrix\n"
            "(descriptive, not a generalization estimate)"
        )

        threshold = matrix.max() / 2

        for row in range(
            matrix.shape[0]
        ):
            for column in range(
                matrix.shape[1]
            ):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column]
                        > threshold
                        else "black"
                    ),
                )

        figure.colorbar(
            image,
            ax=axis,
        )

        figure.tight_layout()

        figure.savefig(
            self.config.output_directory
            / "full_data_confusion_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    # --------------------------------------------------------
    # Validation helpers
    # --------------------------------------------------------

    def _require_loaded_data(self) -> None:
        if (
            self.data is None
            or self.X is None
            or self.y is None
        ):
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

    print("=" * 60)
    print("Classical multinomial GAM with main effects")
    print("=" * 60)

    print(
        "\nMatplotlib backend:",
        matplotlib.get_backend(),
    )

    print(
        "Python executable:",
        sys.executable,
    )

    analysis = ClassicalMultinomialGAM(
        config=config
    )

    analysis.load_data()

    environment_information = (
        analysis.save_environment_information()
    )

    print("\nExecution environment")

    for key, value in environment_information.items():
        print(f"{key}: {value}")

    print("\nRunning data audit...")

    audit = analysis.audit_data()

    print("\nData audit")

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "\nRunning repeated nested cross-validation..."
    )

    fold_results = (
        analysis.evaluate_nested_cv()
    )

    metric_columns = [
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    nested_summary = (
        fold_results[metric_columns]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .T
    )

    print(
        "\nNested cross-validation summary"
    )

    print(
        nested_summary.round(4)
    )

    print(
        "\nHyperparameter selection frequencies"
    )

    for column in [
        "best_n_knots",
        "best_degree",
        "best_C",
    ]:
        print(f"\n{column}")

        print(
            fold_results[column]
            .value_counts()
            .sort_index()
        )

    print(
        "\nFitting the final descriptive model..."
    )

    search = analysis.fit_final_model()

    print(
        "\nBest hyperparameters"
    )

    print(
        search.best_params_
    )

    descriptive_metrics = (
        analysis.full_data_diagnostics()
    )

    print(
        "\nFull-data descriptive metrics"
    )

    print(
        json.dumps(
            descriptive_metrics,
            indent=2,
        )
    )

    print(
        "\nGenerating main-effect plots..."
    )

    analysis.plot_main_effects()

    print(
        "\nAnalysis complete. Results were saved to:"
    )

    print(
        config.output_directory.resolve()
    )


if __name__ == "__main__":
    # Required for reliable multiprocessing behavior on Windows.
    freeze_support()

    main()