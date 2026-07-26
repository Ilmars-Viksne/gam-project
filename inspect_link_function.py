from pathlib import Path

import joblib


MODEL_PATH = Path(
    "outputs/classical_gam_main_effects.joblib"
)


def main() -> None:
    model = joblib.load(MODEL_PATH)

    classifier = model.named_steps["classifier"]

    print("Estimator type:")
    print(type(classifier).__name__)

    print("\nSolver:")
    print(classifier.solver)

    print("\nClasses:")
    print(classifier.classes_)

    print("\nNumber of classes:")
    print(len(classifier.classes_))

    print("\nCoefficient matrix shape:")
    print(classifier.coef_.shape)

    print("\nIntercept vector shape:")
    print(classifier.intercept_.shape)

    print("\nInterpretation:")
    print(
        "Four class-specific scores are transformed into "
        "probabilities by the multinomial softmax function."
    )


if __name__ == "__main__":
    main()