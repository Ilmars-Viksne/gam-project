from __future__ import annotations

import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Non-interactive backend: figures are written directly to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
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
class GAMInteractionConfig:
    """
    Configuration for the multinomial GAM with pairwise
    tensor-product spline interactions.
    """

    data_path: Path = Path("data/dataset.csv")
    output_directory: Path = Path("outputs_pairwise")

    target_column: str = "Y"

    smooth_features: tuple[str, ...] = (
        "X1",
        "X2",
        "X4",
        "X5",
        "X7",
    )

    linear_features: tuple[str, ...] = (
        "X6",
    )

    categorical_features: tuple[str, ...] = (
        "X3",
    )

    class_order: tuple[str, ...] = (
        "O",
        "B",
        "M",
        "G",
    )

    # Default: all ten pairwise combinations among the five
    # smoothly modelled variables.
    interaction_pairs: tuple[tuple[str, str], ...] = (
        ("X1", "X2"),
        ("X1", "X4"),
        ("X1", "X5"),
        ("X1", "X7"),
        ("X2", "X4"),
        ("X2", "X5"),
        ("X2", "X7"),
        ("X4", "X5"),
        ("X4", "X7"),
        ("X5", "X7"),
    )

    random_state: int = 42

    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 5

    # Start conservatively because tensor-product bases grow
    # quadratically with the number of basis functions.
    n_knots_grid: tuple[int, ...] = (
        3,
        4,
        5,
    )

    degree_grid: tuple[int, ...] = (
        2,
        3,
    )

    c_grid: tuple[float, ...] = (
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    )

    # Multiplicative scaling of the interaction columns before
    # regularized logistic regression.
    #
    # Smaller values penalize interactions more strongly relative
    # to the main effects.
    interaction_scale_grid: tuple[float, ...] = (
        0.25,
        0.5,
        1.0,
    )

    # Use 1 during initial testing on Windows. After confirming
    # reliable execution, -1 can use all available CPU workers
    # in the inner GridSearchCV.
    inner_n_jobs: int = 1


# ============================================================
# Tensor-product GAM feature transformer
# ============================================================


class PairwiseSplineFeatures(
    BaseEstimator,
    TransformerMixin,
):
    """
    Generate:

        1. univariate B-spline main-effect columns;
        2. standardized linear main-effect columns;
        3. one-hot categorical main-effect columns;
        4. selected pairwise tensor-product spline columns.

    There are no interactions among already transformed columns
    unless they belong to one of the explicitly specified
    original-variable pairs.

    This prevents accidental products such as two basis functions
    belonging to the same original predictor.
    """

    def __init__(
        self,
        smooth_features: tuple[str, ...],
        linear_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        interaction_pairs: tuple[
            tuple[str, str], ...
        ],
        n_knots: int = 4,
        degree: int = 2,
        interaction_scale: float = 1.0,
    ) -> None:
        self.smooth_features = smooth_features
        self.linear_features = linear_features
        self.categorical_features = categorical_features
        self.interaction_pairs = interaction_pairs

        self.n_knots = n_knots
        self.degree = degree
        self.interaction_scale = interaction_scale

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> PairwiseSplineFeatures:
        X = self._validate_input(X)

        self._validate_interaction_pairs()

        self.spline_transformer_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        )

        self.linear_transformer_ = StandardScaler()

        self.categorical_transformer_ = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )

        smooth_data = X[
            list(self.smooth_features)
        ].astype(float)

        self.spline_transformer_.fit(
            smooth_data
        )

        if self.linear_features:
            self.linear_transformer_.fit(
                X[list(self.linear_features)]
                .astype(float)
            )

        if self.categorical_features:
            self.categorical_transformer_.fit(
                X[list(self.categorical_features)]
                .astype(str)
            )

        number_of_smooth_features = len(
            self.smooth_features
        )

        total_spline_outputs = int(
            self.spline_transformer_.n_features_out_
        )

        if (
            total_spline_outputs
            % number_of_smooth_features
            != 0
        ):
            raise RuntimeError(
                "Spline outputs cannot be divided evenly "
                "among the smooth predictors."
            )

        self.basis_count_per_feature_ = (
            total_spline_outputs
            // number_of_smooth_features
        )

        self.smooth_feature_indices_ = {
            feature: index
            for index, feature
            in enumerate(self.smooth_features)
        }

        self.feature_names_out_ = (
            self._construct_feature_names()
        )

        self.n_features_out_ = len(
            self.feature_names_out_
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        X = self._validate_input(X)

        smooth_data = X[
            list(self.smooth_features)
        ].astype(float)

        smooth_matrix = (
            self.spline_transformer_.transform(
                smooth_data
            )
        )

        blocks: list[np.ndarray] = [
            smooth_matrix
        ]

        if self.linear_features:
            linear_matrix = (
                self.linear_transformer_.transform(
                    X[list(self.linear_features)]
                    .astype(float)
                )
            )

            blocks.append(linear_matrix)

        if self.categorical_features:
            categorical_matrix = (
                self.categorical_transformer_.transform(
                    X[
                        list(
                            self.categorical_features
                        )
                    ].astype(str)
                )
            )

            blocks.append(categorical_matrix)

        for left_feature, right_feature in (
            self.interaction_pairs
        ):
            left_basis = self._extract_spline_block(
                smooth_matrix,
                left_feature,
            )

            right_basis = (
                self._extract_spline_block(
                    smooth_matrix,
                    right_feature,
                )
            )

            # For each row:
            #
            # left_basis[row, :, None]
            #     has shape (q, 1)
            #
            # right_basis[row, None, :]
            #     has shape (1, q)
            #
            # Their product is the q x q tensor-product basis.
            interaction_block = np.einsum(
                "ni,nj->nij",
                left_basis,
                right_basis,
            ).reshape(
                len(X),
                -1,
            )

            interaction_block *= (
                self.interaction_scale
            )

            blocks.append(interaction_block)

        transformed = np.hstack(blocks)

        if transformed.shape[1] != self.n_features_out_:
            raise RuntimeError(
                "Unexpected transformed feature count: "
                f"{transformed.shape[1]} instead of "
                f"{self.n_features_out_}."
            )

        return transformed

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        return np.asarray(
            self.feature_names_out_,
            dtype=object,
        )

    def _extract_spline_block(
        self,
        smooth_matrix: np.ndarray,
        feature_name: str,
    ) -> np.ndarray:
        feature_index = (
            self.smooth_feature_indices_[
                feature_name
            ]
        )

        start = (
            feature_index
            * self.basis_count_per_feature_
        )

        stop = (
            start
            + self.basis_count_per_feature_
        )

        return smooth_matrix[:, start:stop]


    def _construct_feature_names(
        self,
    ) -> list:
        
        names: list[str] = []

        q = self.basis_count_per_feature_

        # Univariate spline main-effect columns.
        for feature_name in self.smooth_features:
            for basis_index in range(q):
                names.append(
                    f"main_spline__{feature_name}"
                    f"__basis_{basis_index}"
                )

        # Standardized linear main-effect columns.
        for feature_name in self.linear_features:
            names.append(
                f"main_linear__{feature_name}"
            )

        # One-hot categorical main-effect columns.
        if self.categorical_features:
            encoded_names = (
                self.categorical_transformer_
                .get_feature_names_out(
                    self.categorical_features
                )
            )

            names.extend(
                f"main_categorical__{name}"
                for name in encoded_names
            )

        # Tensor-product spline interaction columns.
        for left_feature, right_feature in self.interaction_pairs:
            for left_basis_index in range(q):
                for right_basis_index in range(q):
                    names.append(
                        f"interaction__"
                        f"{left_feature}:{right_feature}"
                        f"__basis_"
                        f"{left_basis_index}:"
                        f"{right_basis_index}"
                    )

        return names



    def _validate_input(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PairwiseSplineFeatures expects a "
                "pandas DataFrame."
            )

        required_columns = [
            *self.smooth_features,
            *self.linear_features,
            *self.categorical_features,
        ]

        missing_columns = sorted(
            set(required_columns) - set(X.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing predictor columns: "
                f"{missing_columns}"
            )

        return X

    def _validate_interaction_pairs(
        self,
    ) -> None:
        smooth_feature_set = set(
            self.smooth_features
        )

        normalized_pairs: list[
            tuple[str, str]
        ] = []

        for left_feature, right_feature in (
            self.interaction_pairs
        ):
            if left_feature == right_feature:
                raise ValueError(
                    "An interaction must involve two "
                    "different original predictors."
                )

            if (
                left_feature
                not in smooth_feature_set
                or right_feature
                not in smooth_feature_set
            ):
                raise ValueError(
                    "Tensor-product interactions currently "
                    "support only smooth predictors. Invalid "
                    f"pair: {left_feature}:{right_feature}"
                )

            normalized_pair = tuple(
                sorted(
                    (
                        left_feature,
                        right_feature,
                    )
                )
            )

            if normalized_pair in normalized_pairs:
                raise ValueError(
                    "Duplicate interaction pair: "
                    f"{left_feature}:{right_feature}"
                )

            normalized_pairs.append(
                normalized_pair
            )


# ============================================================
# Pairwise GAM analysis
# ============================================================


class MulticlassPairwiseGAM:
    """
    Repeated nested-CV analysis for the multinomial GAM with
    pairwise tensor-product spline interactions.
    """

    def __init__(
        self,
        config: GAMInteractionConfig,
    ) -> None:
        self.config = config

        self.data: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None

        self.grid_search: GridSearchCV | None = None
        self.best_model: Pipeline | None = None

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    def load_data(self) -> None:
        config = self.config

        if not config.data_path.exists():
            raise FileNotFoundError(
                "CSV file not found:\n"
                f"{config.data_path.resolve()}"
            )

        data = pd.read_csv(
            config.data_path
        )

        required_columns = [
            *config.smooth_features,
            *config.linear_features,
            *config.categorical_features,
            config.target_column,
        ]

        missing_columns = sorted(
            set(required_columns)
            - set(data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        data = data[
            required_columns
        ].copy()

        numeric_columns = [
            *config.smooth_features,
            *config.linear_features,
            *config.categorical_features,
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="raise",
            )

        if data[required_columns].isna().any().any():
            raise ValueError(
                "Missing values were found in the data."
            )

        observed_classes = set(
            data[
                config.target_column
            ].astype(str)
        )

        expected_classes = set(
            config.class_order
        )

        if observed_classes != expected_classes:
            raise ValueError(
                "Observed classes do not match the "
                "configured classes.\n"
                f"Observed: {sorted(observed_classes)}\n"
                f"Expected: {sorted(expected_classes)}"
            )

        # Stable categorical representation.
        for column in config.categorical_features:
            data[column] = data[column].map(
                lambda value: f"{float(value):g}"
            )

        self.data = data

        self.X = data.drop(
            columns=config.target_column
        )

        self.y = data[
            config.target_column
        ].astype(str)

        config.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        configuration = asdict(config)

        configuration["data_path"] = str(
            configuration["data_path"]
        )

        configuration["output_directory"] = str(
            configuration[
                "output_directory"
            ]
        )

        with (
            config.output_directory
            / "analysis_configuration.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                configuration,
                file,
                indent=2,
            )

    # --------------------------------------------------------
    # Pipeline and tuning
    # --------------------------------------------------------

    def build_pipeline(
        self,
    ) -> Pipeline:
        config = self.config

        gam_features = PairwiseSplineFeatures(
            smooth_features=(
                config.smooth_features
            ),
            linear_features=(
                config.linear_features
            ),
            categorical_features=(
                config.categorical_features
            ),
            interaction_pairs=(
                config.interaction_pairs
            ),
            n_knots=4,
            degree=2,
            interaction_scale=0.5,
        )

        classifier = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=20_000,
            class_weight=None,
            random_state=(
                config.random_state
            ),
        )

        return Pipeline(
            steps=[
                (
                    "gam_features",
                    gam_features,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

    def parameter_grid(
        self,
    ) -> dict[str, list[Any]]:
        config = self.config

        return {
            "gam_features__n_knots": list(
                config.n_knots_grid
            ),
            "gam_features__degree": list(
                config.degree_grid
            ),
            "gam_features__interaction_scale": list(
                config.interaction_scale_grid
            ),
            "classifier__C": list(
                config.c_grid
            ),
        }

    def build_grid_search(
        self,
        random_state: int,
        return_train_score: bool = False,
    ) -> GridSearchCV:
        inner_cv = StratifiedKFold(
            n_splits=(
                self.config.inner_splits
            ),
            shuffle=True,
            random_state=random_state,
        )

        return GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.parameter_grid(),
            scoring="neg_log_loss",
            cv=inner_cv,
            refit=True,
            n_jobs=(
                self.config.inner_n_jobs
            ),
            return_train_score=(
                return_train_score
            ),
            error_score="raise",
        )

    # --------------------------------------------------------
    # Manual repeated nested CV
    # --------------------------------------------------------

    def evaluate_nested_cv(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Run repeated nested CV while saving all held-out
        probabilities.

        This is preferable to cross_validate() here because the
        out-of-fold predictions are needed for class-specific
        errors, probability diagnostics, and comparison with the
        main-effects model.
        """

        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config

        outer_cv = RepeatedStratifiedKFold(
            n_splits=config.outer_splits,
            n_repeats=config.outer_repeats,
            random_state=(
                config.random_state
            ),
        )

        fold_rows: list[
            dict[str, Any]
        ] = []

        prediction_frames: list[
            pd.DataFrame
        ] = []

        for outer_iteration, (
            train_indices,
            test_indices,
        ) in enumerate(
            outer_cv.split(
                self.X,
                self.y,
            ),
            start=1,
        ):
            repeat_number = (
                (outer_iteration - 1)
                // config.outer_splits
                + 1
            )

            fold_number = (
                (outer_iteration - 1)
                % config.outer_splits
                + 1
            )

            print(
                "Outer iteration "
                f"{outer_iteration}/"
                f"{config.outer_splits * config.outer_repeats} "
                f"(repeat {repeat_number}, "
                f"fold {fold_number})"
            )

            X_train = self.X.iloc[
                train_indices
            ]

            y_train = self.y.iloc[
                train_indices
            ]

            X_test = self.X.iloc[
                test_indices
            ]

            y_test = self.y.iloc[
                test_indices
            ]

            search = self.build_grid_search(
                random_state=(
                    config.random_state
                    + outer_iteration
                ),
                return_train_score=False,
            )

            search.fit(
                X_train,
                y_train,
            )

            probability = (
                search.predict_proba(
                    X_test
                )
            )

            prediction = search.predict(
                X_test
            )

            class_labels = (
                search.best_estimator_
                .named_steps[
                    "classifier"
                ]
                .classes_
            )

            fold_log_loss = log_loss(
                y_test,
                probability,
                labels=class_labels,
            )

            fold_rows.append(
                {
                    "outer_iteration": (
                        outer_iteration
                    ),
                    "repeat": repeat_number,
                    "fold": fold_number,
                    "test_size": len(
                        test_indices
                    ),
                    "log_loss": float(
                        fold_log_loss
                    ),
                    "accuracy": float(
                        accuracy_score(
                            y_test,
                            prediction,
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            y_test,
                            prediction,
                        )
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_test,
                            prediction,
                            average="macro",
                        )
                    ),
                    "best_n_knots": (
                        search.best_params_[
                            "gam_features__n_knots"
                        ]
                    ),
                    "best_degree": (
                        search.best_params_[
                            "gam_features__degree"
                        ]
                    ),
                    "best_interaction_scale": (
                        search.best_params_[
                            "gam_features__interaction_scale"
                        ]
                    ),
                    "best_C": (
                        search.best_params_[
                            "classifier__C"
                        ]
                    ),
                    "best_inner_log_loss": float(
                        -search.best_score_
                    ),
                }
            )

            prediction_frame = pd.DataFrame(
                {
                    "row_id": (
                        test_indices + 1
                    ),
                    "outer_iteration": (
                        outer_iteration
                    ),
                    "repeat": repeat_number,
                    "fold": fold_number,
                    "observed_class": (
                        y_test.to_numpy()
                    ),
                    "predicted_class": (
                        prediction
                    ),
                }
            )

            for class_index, class_name in (
                enumerate(class_labels)
            ):
                prediction_frame[
                    f"probability_{class_name}"
                ] = probability[
                    :,
                    class_index,
                ]

            prediction_frame[
                "correct"
            ] = (
                prediction_frame[
                    "observed_class"
                ]
                == prediction_frame[
                    "predicted_class"
                ]
            )

            prediction_frames.append(
                prediction_frame
            )

        fold_results = pd.DataFrame(
            fold_rows
        )

        predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        fold_results.to_csv(
            config.output_directory
            / "nested_cv_fold_results.csv",
            index=False,
        )

        predictions.to_csv(
            config.output_directory
            / "nested_cv_predictions.csv",
            index=False,
        )

        self._save_cv_summary(
            fold_results
        )

        self._save_cv_confusion_matrix(
            predictions
        )

        self._save_classification_report(
            predictions
        )

        self._save_hyperparameter_frequencies(
            fold_results
        )

        return fold_results, predictions

    def _save_cv_summary(
        self,
        fold_results: pd.DataFrame,
    ) -> None:
        metrics = [
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]

        summary = (
            fold_results[metrics]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                    "min",
                    "max",
                ]
            )
            .T
        )

        summary[
            "standard_error"
        ] = (
            summary["std"]
            / np.sqrt(
                len(fold_results)
            )
        )

        summary.to_csv(
            self.config.output_directory
            / "nested_cv_summary.csv"
        )

    def _save_cv_confusion_matrix(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        labels = list(
            self.config.class_order
        )

        matrix = confusion_matrix(
            predictions[
                "observed_class"
            ],
            predictions[
                "predicted_class"
            ],
            labels=labels,
        )

        matrix_frame = pd.DataFrame(
            matrix,
            index=[
                f"Observed_{label}"
                for label in labels
            ],
            columns=[
                f"Predicted_{label}"
                for label in labels
            ],
        )

        matrix_frame.to_csv(
            self.config.output_directory
            / "nested_cv_confusion_matrix.csv"
        )

        figure, axis = plt.subplots(
            figsize=(6.5, 5.5)
        )

        image = axis.imshow(matrix)

        axis.set_xticks(
            np.arange(len(labels))
        )

        axis.set_yticks(
            np.arange(len(labels))
        )

        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)

        axis.set_xlabel(
            "Predicted class"
        )

        axis.set_ylabel(
            "Observed class"
        )

        axis.set_title(
            "Repeated nested-CV confusion matrix"
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
                    str(
                        matrix[
                            row,
                            column,
                        ]
                    ),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[
                            row,
                            column
                        ] > threshold
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
            / "nested_cv_confusion_matrix.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

    def _save_classification_report(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        report = classification_report(
            predictions[
                "observed_class"
            ],
            predictions[
                "predicted_class"
            ],
            labels=list(
                self.config.class_order
            ),
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).T.to_csv(
            self.config.output_directory
            / "nested_cv_classification_report.csv"
        )

    def _save_hyperparameter_frequencies(
        self,
        fold_results: pd.DataFrame,
    ) -> None:
        parameter_columns = [
            "best_n_knots",
            "best_degree",
            "best_interaction_scale",
            "best_C",
        ]

        frames: list[pd.DataFrame] = []

        for parameter in parameter_columns:
            frequencies = (
                fold_results[parameter]
                .value_counts()
                .sort_index()
                .rename_axis("value")
                .reset_index(name="count")
            )

            frequencies[
                "proportion"
            ] = (
                frequencies["count"]
                / len(fold_results)
            )

            frequencies.insert(
                0,
                "parameter",
                parameter,
            )

            frames.append(
                frequencies
            )

        pd.concat(
            frames,
            ignore_index=True,
        ).to_csv(
            self.config.output_directory
            / "nested_cv_hyperparameter_frequencies.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    def fit_final_model(
        self,
    ) -> GridSearchCV:
        self._require_loaded_data()

        assert self.X is not None
        assert self.y is not None

        config = self.config

        self.grid_search = (
            self.build_grid_search(
                random_state=(
                    config.random_state
                ),
                return_train_score=True,
            )
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
            / "classical_gam_pairwise.joblib",
        )

        best_parameters = {
            key: (
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else value
            )
            for key, value
            in self.grid_search
            .best_params_
            .items()
        }

        best_parameters[
            "best_inner_cv_log_loss"
        ] = float(
            -self.grid_search.best_score_
        )

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

        self._save_final_coefficients()

        return self.grid_search

    def _save_final_coefficients(
        self,
    ) -> None:
        self._require_fitted_model()

        assert self.best_model is not None

        transformer = (
            self.best_model.named_steps[
                "gam_features"
            ]
        )

        classifier = (
            self.best_model.named_steps[
                "classifier"
            ]
        )

        feature_names = (
            transformer
            .get_feature_names_out()
        )

        coefficients = pd.DataFrame(
            classifier.coef_.T,
            columns=[
                f"coefficient_class_{label}"
                for label
                in classifier.classes_
            ],
        )

        coefficients.insert(
            0,
            "term",
            feature_names,
        )

        coefficients[
            "term_type"
        ] = np.where(
            coefficients["term"]
            .str.startswith(
                "interaction__"
            ),
            "interaction",
            "main_effect",
        )

        coefficients.to_csv(
            self.config.output_directory
            / "final_model_coefficients.csv",
            index=False,
        )

        interaction_coefficients = (
            coefficients.loc[
                coefficients[
                    "term_type"
                ]
                == "interaction"
            ]
            .copy()
        )

        interaction_coefficients[
            "interaction_pair"
        ] = (
            interaction_coefficients[
                "term"
            ]
            .str.extract(
                r"interaction__([^_]+)__"
            )
        )

        coefficient_columns = [
            column
            for column
            in interaction_coefficients.columns
            if column.startswith(
                "coefficient_class_"
            )
        ]

        interaction_coefficients[
            "coefficient_l2_norm"
        ] = np.sqrt(
            np.square(
                interaction_coefficients[
                    coefficient_columns
                ]
            ).sum(axis=1)
        )

        pair_importance = (
            interaction_coefficients
            .groupby(
                "interaction_pair",
                as_index=False,
            )[
                "coefficient_l2_norm"
            ]
            .agg(
                total_coefficient_norm="sum",
                mean_coefficient_norm="mean",
                maximum_coefficient_norm="max",
            )
            .sort_values(
                "total_coefficient_norm",
                ascending=False,
            )
        )

        pair_importance.to_csv(
            self.config.output_directory
            / "interaction_coefficient_ranking.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Final descriptive diagnostics
    # --------------------------------------------------------

    def full_data_diagnostics(
        self,
    ) -> dict[str, float]:
        self._require_fitted_model()

        assert self.best_model is not None
        assert self.X is not None
        assert self.y is not None

        probabilities = (
            self.best_model.predict_proba(
                self.X
            )
        )

        predictions = (
            self.best_model.predict(
                self.X
            )
        )

        labels = (
            self.best_model.named_steps[
                "classifier"
            ].classes_
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
                )
            ),
            "multiclass_log_loss": float(
                log_loss(
                    self.y,
                    probabilities,
                    labels=labels,
                )
            ),
            "maximum_probability_sum_error": float(
                np.abs(
                    probabilities.sum(
                        axis=1
                    )
                    - 1.0
                ).max()
            ),
        }

        with (
            self.config.output_directory
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

        return metrics

    # --------------------------------------------------------
    # Interaction surface plots
    # --------------------------------------------------------

    def plot_interaction_surfaces(
        self,
        grid_size: int = 40,
    ) -> None:
        """
        Plot the pure fitted tensor-product contribution for each
        configured pair and class.

        Main effects, intercepts, linear terms, and categorical
        terms are not included in these surfaces.
        """

        self._require_fitted_model()

        assert self.best_model is not None
        assert self.X is not None

        model = self.best_model

        transformer = model.named_steps[
            "gam_features"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        q = (
            transformer
            .basis_count_per_feature_
        )

        number_of_smooth_main_columns = (
            len(
                transformer.smooth_features
            )
            * q
        )

        number_of_linear_columns = len(
            transformer.linear_features
        )

        if transformer.categorical_features:
            number_of_categorical_columns = int(
                transformer
                .categorical_transformer_
                .transform(
                    self.X[
                        list(
                            transformer
                            .categorical_features
                        )
                    ].iloc[:1].astype(str)
                )
                .shape[1]
            )
        else:
            number_of_categorical_columns = 0

        interaction_start = (
            number_of_smooth_main_columns
            + number_of_linear_columns
            + number_of_categorical_columns
        )

        pair_block_size = q * q

        smooth_reference = {
            feature: float(
                self.X[feature].median()
            )
            for feature
            in transformer.smooth_features
        }

        for pair_index, (
            left_feature,
            right_feature,
        ) in enumerate(
            transformer.interaction_pairs
        ):
            left_values = (
                self.X[left_feature]
                .astype(float)
                .to_numpy()
            )

            right_values = (
                self.X[right_feature]
                .astype(float)
                .to_numpy()
            )

            left_grid = np.linspace(
                np.quantile(
                    left_values,
                    0.02,
                ),
                np.quantile(
                    left_values,
                    0.98,
                ),
                grid_size,
            )

            right_grid = np.linspace(
                np.quantile(
                    right_values,
                    0.02,
                ),
                np.quantile(
                    right_values,
                    0.98,
                ),
                grid_size,
            )

            left_mesh, right_mesh = (
                np.meshgrid(
                    left_grid,
                    right_grid,
                )
            )

            grid_rows = (
                grid_size * grid_size
            )

            smooth_grid = pd.DataFrame(
                {
                    feature: np.full(
                        grid_rows,
                        smooth_reference[
                            feature
                        ],
                    )
                    for feature
                    in transformer.smooth_features
                }
            )

            smooth_grid[
                left_feature
            ] = left_mesh.ravel()

            smooth_grid[
                right_feature
            ] = right_mesh.ravel()

            spline_matrix = (
                transformer
                .spline_transformer_
                .transform(
                    smooth_grid
                )
            )

            left_index = (
                transformer
                .smooth_feature_indices_[
                    left_feature
                ]
            )

            right_index = (
                transformer
                .smooth_feature_indices_[
                    right_feature
                ]
            )

            left_basis = spline_matrix[
                :,
                left_index * q:
                (left_index + 1) * q,
            ]

            right_basis = spline_matrix[
                :,
                right_index * q:
                (right_index + 1) * q,
            ]

            interaction_basis = np.einsum(
                "ni,nj->nij",
                left_basis,
                right_basis,
            ).reshape(
                grid_rows,
                -1,
            )

            interaction_basis *= (
                transformer
                .interaction_scale
            )

            block_start = (
                interaction_start
                + pair_index
                * pair_block_size
            )

            block_stop = (
                block_start
                + pair_block_size
            )

            pair_coefficients = (
                classifier.coef_[
                    :,
                    block_start:block_stop,
                ]
            )

            for class_index, class_name in enumerate(
                classifier.classes_
            ):
                surface = (
                    interaction_basis
                    @ pair_coefficients[
                        class_index
                    ]
                ).reshape(
                    grid_size,
                    grid_size,
                )

                # Center for interpretability.
                surface -= surface.mean()

                figure, axis = plt.subplots(
                    figsize=(8, 6)
                )

                contour = axis.contourf(
                    left_mesh,
                    right_mesh,
                    surface,
                    levels=20,
                )

                axis.set_xlabel(
                    left_feature
                )

                axis.set_ylabel(
                    right_feature
                )

                axis.set_title(
                    "Pairwise interaction contribution: "
                    f"{left_feature} × {right_feature}, "
                    f"class {class_name}"
                )

                figure.colorbar(
                    contour,
                    ax=axis,
                    label=(
                        "Centered additive "
                        "score contribution"
                    ),
                )

                figure.tight_layout()

                figure.savefig(
                    self.config.output_directory
                    / (
                        "interaction_"
                        f"{left_feature}_"
                        f"{right_feature}_"
                        f"class_{class_name}.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )

                plt.close(figure)

    # --------------------------------------------------------
    # State checks
    # --------------------------------------------------------

    def _require_loaded_data(
        self,
    ) -> None:
        if (
            self.data is None
            or self.X is None
            or self.y is None
        ):
            raise RuntimeError(
                "Call load_data() first."
            )

    def _require_fitted_model(
        self,
    ) -> None:
        self._require_loaded_data()

        if self.best_model is None:
            raise RuntimeError(
                "Call fit_final_model() first."
            )


# ============================================================
# Main
# ============================================================


def main() -> None:
    config = GAMInteractionConfig()

    analysis = MulticlassPairwiseGAM(
        config
    )

    print(
        "\nLoading data from:"
        f"\n{config.data_path.resolve()}"
    )

    analysis.load_data()

    print("\nPairwise interactions")

    for left_feature, right_feature in (
        config.interaction_pairs
    ):
        print(
            f"  {left_feature} × "
            f"{right_feature}"
        )

    print(
        "\nRunning repeated nested "
        "cross-validation..."
    )

    fold_results, predictions = (
        analysis.evaluate_nested_cv()
    )

    metrics = [
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]

    print(
        "\nNested cross-validation summary"
    )

    print(
        fold_results[metrics]
        .agg(["mean", "std"])
        .T
        .round(4)
    )

    print(
        "\nSelected hyperparameters"
    )

    for parameter in (
        "best_n_knots",
        "best_degree",
        "best_interaction_scale",
        "best_C",
    ):
        print(f"\n{parameter}")

        print(
            fold_results[parameter]
            .value_counts()
            .sort_index()
        )

    print(
        "\nFitting the final descriptive "
        "pairwise model..."
    )

    search = analysis.fit_final_model()

    print("\nBest hyperparameters")

    print(
        json.dumps(
            {
                key: (
                    value.item()
                    if isinstance(
                        value,
                        np.generic,
                    )
                    else value
                )
                for key, value
                in search.best_params_.items()
            },
            indent=2,
        )
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
        "\nGenerating interaction surfaces..."
    )

    analysis.plot_interaction_surfaces()

    print(
        "\nAnalysis complete. Results saved to:"
        f"\n{config.output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()