from __future__ import annotations

import __main__
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Model configurations
# ============================================================

MODEL_CONFIGURATIONS = {
    "classic": {
        "model_path": Path(
            "outputs/classical_gam_main_effects.joblib"
        ),
        "output_path": Path(
            "outputs/final_gam_components.csv"
        ),
        "equations_path": Path(
            "outputs/final_gam_equations.txt"
        ),
        "intercepts_path": Path(
            "outputs/final_gam_intercepts.csv"
        ),
        "reference_equations_path": Path(
            "outputs/reference_class_link_equations.csv"
        ),
        "transformer_step": "preprocessor",
    },
    "pairwise": {
        "model_path": Path(
            "outputs_pairwise/classical_gam_pairwise.joblib"
        ),
        "output_path": Path(
            "outputs_pairwise/final_gam_components.csv"
        ),
        "equations_path": Path(
            "outputs_pairwise/final_gam_equations.txt"
        ),
        "intercepts_path": Path(
            "outputs_pairwise/final_gam_intercepts.csv"
        ),
        "reference_equations_path": Path(
            "outputs_pairwise/reference_class_link_equations.csv"
        ),
        "transformer_step": "gam_features",
    },
}


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """
    Read the model type from the command line.

    Examples
    --------
    python inspect_final_gam.py classic

    python inspect_final_gam.py pairwise
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect a fitted classical main-effects GAM "
            "or pairwise-interaction GAM."
        )
    )

    parser.add_argument(
        "model_type",
        choices=(
            "classic",
            "pairwise",
        ),
        help=(
            "Select 'classic' for the main-effects GAM or "
            "'pairwise' for the pairwise-interaction GAM."
        ),
    )

    parser.add_argument(
        "--reference-class",
        default="O",
        help=(
            "Reference class used when exporting conventional "
            "multinomial-logit equations. Default: O."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help=(
            "Number of largest coefficients to print for each "
            "class. Default: 15."
        ),
    )

    return parser.parse_args()


# ============================================================
# Pairwise-model compatibility fix
# ============================================================


def register_pairwise_compatibility_class() -> None:
    """
    Register PairwiseSplineFeatures under __main__.

    The existing pairwise model was saved while
    classical_gam_pairwise.py was executed directly. Python
    therefore serialized its custom transformer as:

        __main__.PairwiseSplineFeatures

    When this inspection file is executed, inspect_final_gam.py
    becomes the new __main__ module. The alias below makes the
    original serialized class path available before joblib.load()
    is called.
    """

    try:
        from classical_gam_pairwise import (
            PairwiseSplineFeatures,
        )
    except ImportError as error:
        raise ImportError(
            "Unable to import PairwiseSplineFeatures from "
            "classical_gam_pairwise.py.\n\n"
            "Ensure that:\n"
            "1. classical_gam_pairwise.py is in the same folder "
            "as inspect_final_gam.py;\n"
            "2. PairwiseSplineFeatures is defined in that file;\n"
            "3. classical_gam_pairwise.py passes the command:\n"
            "   python -m py_compile classical_gam_pairwise.py"
        ) from error

    __main__.PairwiseSplineFeatures = (
        PairwiseSplineFeatures
    )


# ============================================================
# Model loading and validation
# ============================================================


def load_model(
    model_type: str,
    model_path: Path,
) -> Any:
    """
    Load the requested model.

    The compatibility class is registered only when the pairwise
    model is requested.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            "The fitted model file was not found:\n"
            f"{model_path.resolve()}"
        )

    if model_type == "pairwise":
        register_pairwise_compatibility_class()

    try:
        model = joblib.load(
            model_path
        )
    except AttributeError as error:
        if model_type == "pairwise":
            raise RuntimeError(
                "The pairwise model could not be loaded because "
                "a serialized custom class was not found.\n\n"
                "The compatibility alias was registered, but "
                "the saved file may reference another custom "
                "class or a differently named transformer.\n\n"
                f"Original error:\n{error}"
            ) from error

        raise

    if not hasattr(
        model,
        "named_steps",
    ):
        raise TypeError(
            "The loaded object is not a fitted scikit-learn "
            "Pipeline because it has no named_steps attribute."
        )

    return model


def get_model_objects(
    model: Any,
    transformer_step: str,
) -> tuple[Any, Any, np.ndarray]:
    """
    Retrieve the fitted feature transformer, classifier, and
    transformed feature names.
    """

    available_steps = list(
        model.named_steps.keys()
    )

    if transformer_step not in model.named_steps:
        raise KeyError(
            f"The expected transformer step "
            f"{transformer_step!r} was not found.\n"
            f"Available pipeline steps: {available_steps}"
        )

    if "classifier" not in model.named_steps:
        raise KeyError(
            "The pipeline does not contain a 'classifier' step.\n"
            f"Available pipeline steps: {available_steps}"
        )

    transformer = model.named_steps[
        transformer_step
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    if not hasattr(
        transformer,
        "get_feature_names_out",
    ):
        raise AttributeError(
            f"The transformer {type(transformer).__name__!r} "
            "does not implement get_feature_names_out()."
        )

    feature_names = np.asarray(
        transformer.get_feature_names_out(),
        dtype=object,
    )

    coefficient_count = (
        classifier.coef_.shape[1]
    )

    if len(feature_names) != coefficient_count:
        raise RuntimeError(
            "The number of transformed feature names does not "
            "match the number of classifier coefficients.\n"
            f"Feature names: {len(feature_names)}\n"
            f"Coefficients per class: {coefficient_count}"
        )

    return (
        transformer,
        classifier,
        feature_names,
    )


# ============================================================
# Term classification
# ============================================================


def determine_component_type(
    component_name: str,
) -> str:
    """
    Assign each transformed column to a component type.

    Supports names produced by both the classical model's
    ColumnTransformer and the custom pairwise transformer.
    """

    name = str(
        component_name
    )

    # Pairwise-transformer naming convention.
    if name.startswith(
        "main_spline__"
    ):
        return "smooth_main_effect"

    if name.startswith(
        "main_linear__"
    ):
        return "linear_main_effect"

    if name.startswith(
        "main_categorical__"
    ):
        return "categorical_main_effect"

    if name.startswith(
        "interaction__"
    ):
        return "pairwise_interaction"

    # Classical ColumnTransformer naming convention.
    if name.startswith(
        "smooth__"
    ):
        return "smooth_main_effect"

    if name.startswith(
        "linear__"
    ):
        return "linear_main_effect"

    if name.startswith(
        "categorical__"
    ):
        return "categorical_main_effect"

    return "unclassified"


def extract_original_component(
    component_name: str,
) -> str:
    """
    Extract the original predictor or predictor pair from a
    transformed feature name.
    """

    name = str(
        component_name
    )

    if name.startswith(
        "interaction__"
    ):
        remainder = name.removeprefix(
            "interaction__"
        )

        return remainder.split(
            "__basis_",
            maxsplit=1,
        )[0]

    if name.startswith(
        "main_spline__"
    ):
        remainder = name.removeprefix(
            "main_spline__"
        )

        return remainder.split(
            "__basis_",
            maxsplit=1,
        )[0]

    if name.startswith(
        "main_linear__"
    ):
        return name.removeprefix(
            "main_linear__"
        )

    if name.startswith(
        "main_categorical__"
    ):
        return name.removeprefix(
            "main_categorical__"
        )

    if name.startswith(
        "smooth__"
    ):
        remainder = name.removeprefix(
            "smooth__"
        )

        # Classical SplineTransformer names may resemble:
        # X1_sp_0
        return remainder.split(
            "_sp_",
            maxsplit=1,
        )[0]

    if name.startswith(
        "linear__"
    ):
        return name.removeprefix(
            "linear__"
        )

    if name.startswith(
        "categorical__"
    ):
        return name.removeprefix(
            "categorical__"
        )

    return name


# ============================================================
# Symbolic transformed-space equations
# ============================================================


def build_symbolic_equations(
    classifier: Any,
    feature_names: np.ndarray,
) -> str:
    """
    Construct the complete transformed-space score equation for
    every fitted class.
    """

    lines: list[str] = []

    lines.extend(
        [
            "MULTICLASS GAM SCORE EQUATIONS",
            "=" * 78,
            "",
            "Each equation defines a class-specific additive score.",
            "The class probabilities are obtained using softmax:",
            "",
            "    P(Y = k | x) = exp(eta_k) / sum_l exp(eta_l)",
            "",
        ]
    )

    for class_index, class_name in enumerate(
        classifier.classes_
    ):
        intercept = float(
            classifier.intercept_[
                class_index
            ]
        )

        lines.append(
            "=" * 78
        )

        lines.append(
            f"Score equation for class {class_name}"
        )

        lines.append(
            "=" * 78
        )

        lines.append(
            f"eta_{class_name}(x) = "
            f"{intercept:.10f}"
        )

        for feature_name, coefficient in zip(
            feature_names,
            classifier.coef_[
                class_index
            ],
            strict=True,
        ):
            coefficient = float(
                coefficient
            )

            sign = (
                "+"
                if coefficient >= 0
                else "-"
            )

            lines.append(
                f"    {sign} "
                f"{abs(coefficient):.10f}"
                f" * {feature_name}"
            )

        lines.append("")

    return "\n".join(
        lines
    )


def print_symbolic_equations(
    classifier: Any,
    feature_names: np.ndarray,
    equations_path: Path,
) -> None:
    """
    Print and save all transformed-space class-score equations.
    """

    equation_text = build_symbolic_equations(
        classifier=classifier,
        feature_names=feature_names,
    )

    print(
        "\n" + equation_text
    )

    equations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    equations_path.write_text(
        equation_text,
        encoding="utf-8",
    )


# ============================================================
# Component export
# ============================================================


def create_component_table(
    classifier: Any,
    feature_names: np.ndarray,
) -> pd.DataFrame:
    """
    Create a tidy table containing every fitted coefficient.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for class_index, class_name in enumerate(
        classifier.classes_
    ):
        for feature_index, feature_name in enumerate(
            feature_names
        ):
            coefficient = float(
                classifier.coef_[
                    class_index,
                    feature_index,
                ]
            )

            rows.append(
                {
                    "class": str(
                        class_name
                    ),
                    "component_type": (
                        determine_component_type(
                            str(feature_name)
                        )
                    ),
                    "original_component": (
                        extract_original_component(
                            str(feature_name)
                        )
                    ),
                    "component": str(
                        feature_name
                    ),
                    "coefficient": (
                        coefficient
                    ),
                    "absolute_coefficient": abs(
                        coefficient
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def save_intercepts(
    classifier: Any,
    intercepts_path: Path,
) -> pd.DataFrame:
    """
    Save the fitted class-specific intercept vector.
    """

    intercepts = pd.DataFrame(
        {
            "class": [
                str(class_name)
                for class_name
                in classifier.classes_
            ],
            "intercept": (
                classifier.intercept_
                .astype(float)
            ),
        }
    )

    intercepts_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    intercepts.to_csv(
        intercepts_path,
        index=False,
    )

    return intercepts


# ============================================================
# Reference-class link equations
# ============================================================


def create_reference_class_equations(
    classifier: Any,
    feature_names: np.ndarray,
    reference_class: str,
) -> pd.DataFrame:
    """
    Convert the class-score representation into conventional
    multinomial-logit equations relative to one reference class.

    For class k and reference class r:

        log(P(Y=k|x) / P(Y=r|x))
        =
        eta_k(x) - eta_r(x)
    """

    classes = [
        str(class_name)
        for class_name
        in classifier.classes_
    ]

    if reference_class not in classes:
        raise ValueError(
            f"Reference class {reference_class!r} is not among "
            f"the fitted classes: {classes}"
        )

    reference_index = classes.index(
        reference_class
    )

    rows: list[
        dict[str, Any]
    ] = []

    for class_index, class_name in enumerate(
        classes
    ):
        if class_name == reference_class:
            continue

        comparison = (
            f"{class_name}_versus_"
            f"{reference_class}"
        )

        intercept_difference = float(
            classifier.intercept_[
                class_index
            ]
            - classifier.intercept_[
                reference_index
            ]
        )

        rows.append(
            {
                "comparison": comparison,
                "component_type": "intercept",
                "original_component": "intercept",
                "component": "intercept",
                "coefficient": (
                    intercept_difference
                ),
                "odds_ratio": np.nan,
            }
        )

        coefficient_differences = (
            classifier.coef_[
                class_index
            ]
            - classifier.coef_[
                reference_index
            ]
        )

        for feature_name, coefficient in zip(
            feature_names,
            coefficient_differences,
            strict=True,
        ):
            coefficient = float(
                coefficient
            )

            # Exponentiating a single spline-basis coefficient
            # is mathematically possible but usually not useful
            # as an isolated scientific odds ratio. It is retained
            # here as an exact transformed-space quantity.
            odds_ratio = float(
                np.exp(
                    np.clip(
                        coefficient,
                        -700.0,
                        700.0,
                    )
                )
            )

            rows.append(
                {
                    "comparison": comparison,
                    "component_type": (
                        determine_component_type(
                            str(feature_name)
                        )
                    ),
                    "original_component": (
                        extract_original_component(
                            str(feature_name)
                        )
                    ),
                    "component": str(
                        feature_name
                    ),
                    "coefficient": (
                        coefficient
                    ),
                    "odds_ratio": (
                        odds_ratio
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Console summaries
# ============================================================


def print_model_information(
    model_type: str,
    model_path: Path,
    transformer_step: str,
    transformer: Any,
    classifier: Any,
    feature_names: np.ndarray,
) -> None:
    """
    Print basic structural information about the fitted pipeline.
    """

    print("\n" + "=" * 78)
    print("FITTED MODEL INFORMATION")
    print("=" * 78)

    print(
        f"Model type: {model_type}"
    )

    print(
        f"Model path: {model_path.resolve()}"
    )

    print(
        f"Transformer step: {transformer_step}"
    )

    print(
        "Transformer type: "
        f"{type(transformer).__module__}."
        f"{type(transformer).__name__}"
    )

    print(
        "Classifier type: "
        f"{type(classifier).__module__}."
        f"{type(classifier).__name__}"
    )

    print(
        f"Solver: {classifier.solver}"
    )

    print(
        f"Regularization C: {classifier.C}"
    )

    print(
        "Classes: "
        f"{list(classifier.classes_)}"
    )

    print(
        "Coefficient matrix shape: "
        f"{classifier.coef_.shape}"
    )

    print(
        "Number of transformed components: "
        f"{len(feature_names)}"
    )

    print(
        "\nResponse/link structure:"
    )

    print(
        "  Response distribution: multinomial"
    )

    print(
        "  Link: multinomial logit"
    )

    print(
        "  Inverse link: softmax"
    )

def print_intercepts(
    intercepts: pd.DataFrame,
) -> None:
    """
    Print class-specific intercept values.
    """

    print("\nClasses")

    print(
        intercepts["class"].to_numpy()
    )

    print("\nIntercepts")

    for row in intercepts.itertuples(
        index=False,
        name=None,
    ):
        class_name, intercept = row

        print(
            f"eta_{class_name}: "
            f"intercept = {intercept:.10f}"
        )

def print_largest_coefficients(
    components: pd.DataFrame,
    number_to_show: int,
) -> None:
    """
    Print the largest absolute transformed coefficients for every
    class.
    """

    for class_name in components[
        "class"
    ].unique():
        class_components = (
            components.loc[
                components["class"]
                == class_name
            ]
            .sort_values(
                "absolute_coefficient",
                ascending=False,
            )
        )

        print(
            f"\nLargest transformed coefficients "
            f"for class {class_name}"
        )

        print(
            class_components[
                [
                    "component_type",
                    "original_component",
                    "component",
                    "coefficient",
                ]
            ]
            .head(number_to_show)
            .to_string(index=False)
        )


def print_component_counts(
    components: pd.DataFrame,
) -> None:
    """
    Print transformed-component counts for one class.

    Every class has the same transformed design columns, so only
    one class is needed for this count.
    """

    first_class = components[
        "class"
    ].iloc[0]

    counts = (
        components.loc[
            components["class"]
            == first_class,
            "component_type",
        ]
        .value_counts()
    )

    print(
        "\nTransformed component counts per class"
    )

    print(
        counts.to_string()
    )


# ============================================================
# Main execution
# ============================================================


def main() -> None:
    arguments = parse_arguments()

    model_type = (
        arguments.model_type
    )

    configuration = (
        MODEL_CONFIGURATIONS[
            model_type
        ]
    )

    model_path = configuration[
        "model_path"
    ]

    output_path = configuration[
        "output_path"
    ]

    equations_path = configuration[
        "equations_path"
    ]

    intercepts_path = configuration[
        "intercepts_path"
    ]

    reference_equations_path = (
        configuration[
            "reference_equations_path"
        ]
    )

    transformer_step = configuration[
        "transformer_step"
    ]

    model = load_model(
        model_type=model_type,
        model_path=model_path,
    )

    (
        transformer,
        classifier,
        feature_names,
    ) = get_model_objects(
        model=model,
        transformer_step=transformer_step,
    )

    print_model_information(
        model_type=model_type,
        model_path=model_path,
        transformer_step=transformer_step,
        transformer=transformer,
        classifier=classifier,
        feature_names=feature_names,
    )

    print_symbolic_equations(
        classifier=classifier,
        feature_names=feature_names,
        equations_path=equations_path,
    )

    components = create_component_table(
        classifier=classifier,
        feature_names=feature_names,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    components.to_csv(
        output_path,
        index=False,
    )

    intercepts = save_intercepts(
        classifier=classifier,
        intercepts_path=intercepts_path,
    )

    reference_equations = (
        create_reference_class_equations(
            classifier=classifier,
            feature_names=feature_names,
            reference_class=(
                arguments.reference_class
            ),
        )
    )

    reference_equations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_equations.to_csv(
        reference_equations_path,
        index=False,
    )

    print_intercepts(
        intercepts=intercepts
    )

    print_component_counts(
        components=components
    )

    print_largest_coefficients(
        components=components,
        number_to_show=arguments.top,
    )

    print(
        "\nAll components were saved to:"
    )

    print(
        output_path.resolve()
    )

    print(
        "\nSymbolic transformed-space equations "
        "were saved to:"
    )

    print(
        equations_path.resolve()
    )

    print(
        "\nClass intercepts were saved to:"
    )

    print(
        intercepts_path.resolve()
    )

    print(
        "\nReference-class multinomial-logit "
        "equations were saved to:"
    )

    print(
        reference_equations_path.resolve()
    )


if __name__ == "__main__":
    main()