# Multiclass Generalized Additive Models with Pairwise Interactions

A Python research project for modelling the four-class response `Y ∈ {O, B, M, G}` using interpretable multinomial additive models.

The project compares two classical GAM-style specifications:

1. **Main-effects multiclass GAM** — univariate spline effects, one linear effect, and one categorical effect.
2. **Multiclass GAM with pairwise interactions** — the same main effects plus selected tensor-product spline interactions.

The primary research question is not whether a more complex model can fit the observed data better. It is whether pairwise interactions provide **reproducible improvements on unseen observations** while remaining stable, supported by the data, and interpretable.

---

## Contents

- [Research objective](#research-objective)
- [Dataset](#dataset)
- [Model definitions](#model-definitions)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running the analysis](#running-the-analysis)
- [Using comparison.py](#using-comparisonpy)
- [Using inspect_final_gam.py](#using-inspect_final_gampy)
- [Using inspect_link_function.py](#using-inspect_link_functionpy)
- [Main-effects model](#main-effects-model)
- [Pairwise-interaction model](#pairwise-interaction-model)
- [Choosing interaction pairs](#choosing-interaction-pairs)
- [Nested cross-validation](#nested-cross-validation)
- [Model comparison](#model-comparison)
- [Interpreting outputs](#interpreting-outputs)
- [Known data considerations](#known-data-considerations)
- [Windows and Matplotlib notes](#windows-and-matplotlib-notes)
- [Computational cost](#computational-cost)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Recommended research workflow](#recommended-research-workflow)

---

## Research objective

Let the predictors be

$$
\mathbf{x}=(X_1,X_2,X_3,X_4,X_5,X_6,X_7),
$$

and let the response contain four unordered classes:

$$
Y\in\{O,B,M,G\}.
$$

The project evaluates whether class membership can be modelled accurately and interpretably using additive spline functions, and whether pairwise interactions improve generalization beyond a main-effects-only model.

The comparison is organized around these questions:

1. How well does a penalized multinomial additive spline model generalize?
2. Do tensor-product spline interactions improve multiclass log loss?
3. Which interactions, if any, are repeatedly selected across resamples?
4. Are selected interaction surfaces supported by observed predictor combinations?
5. Does the increased complexity provide enough benefit to justify reduced parsimony?

The primary evaluation metric is **multiclass log loss**. Accuracy, balanced accuracy, and macro-F1 are secondary metrics.

---

## Dataset

The expected CSV schema is:

```text
X1,X2,X3,X4,X5,X6,X7,Y
```

The current dataset contains:

- 500 observations;
- 7 predictors;
- 4 response classes;
- no missing values in the supplied data;
- 2 exact duplicate predictor groups involving 4 rows;
- no duplicate predictor configurations with conflicting labels;
- 10 observations with `X5 > 40`.

Observed class distribution:

| Class | Count | Proportion |
|---|---:|---:|
| O | 237 | 0.474 |
| B | 87 | 0.174 |
| M | 106 | 0.212 |
| G | 70 | 0.140 |

Observed numbers of unique predictor values:

| Predictor | Unique values |
|---|---:|
| X1 | 61 |
| X2 | 56 |
| X3 | 3 |
| X4 | 44 |
| X5 | 18 |
| X6 | 3 |
| X7 | 6 |

### Current predictor representation

The initial model uses the following representation:

- **Smooth spline effects:** `X1`, `X2`, `X4`, `X5`, `X7`
- **Standardized linear effect:** `X6`
- **Categorical effect:** `X3`

`X3` has only three observed values and is represented categorically. `X6` also has three values, but it is treated as an ordered numerical measurement with a linear effect. This avoids unknown-category warnings when a rare `X6` value is absent from an inner training fold.

---

## Model definitions

### Main-effects multiclass GAM

For class $k$, the class-specific score is

$$
\eta_k(\mathbf{x})=
\beta_{0k}
+f_{1k}(X_1)
+f_{2k}(X_2)
+f_{4k}(X_4)
+f_{5k}(X_5)
+f_{7k}(X_7)
+\beta_{6k}\widetilde{X}_6
+\gamma_k(X_3),
$$

where:

- each $f_{jk}$ is a univariate B-spline expansion;
- $\widetilde{X}_6$ is standardized `X6`;
- $\gamma_k(X_3)$ is a categorical main effect;
- no predictor products or pairwise surfaces are included.

Class probabilities are produced by the softmax transformation:

$$
P(Y=k\mid\mathbf{x})=
\frac{\exp(\eta_k(\mathbf{x}))}
{\sum_\ell \exp(\eta_\ell(\mathbf{x}))}.
$$

The implementation is most precisely described as a **penalized multinomial logistic additive B-spline model**. L2 regularization is applied to the resulting spline and main-effect coefficients.

### Multiclass GAM with pairwise interactions

The interaction model extends the class score to

$$
\eta_k(\mathbf{x})=
\beta_{0k}
+\sum_j f_{jk}(x_j)
+\sum_{(r,s)\in\mathcal I} f_{rsk}(x_r,x_s).
$$

For a selected pair $(X_r,X_s)$, the interaction is represented using a tensor-product spline basis:

$$
f_{rsk}(x_r,x_s)=
\sum_a\sum_b
\theta_{krsab} B_{ra}(x_r)B_{sb}(x_s).
$$

All component main effects remain in the model. This follows the hierarchy principle: an interaction is not fitted without its corresponding main effects.

### Link function and probability model

Both model specifications use the same response and link structure:

- **response distribution:** multinomial;
- **link:** multinomial logit;
- **inverse link:** softmax;
- **optimization objective:** L2-penalized multinomial log loss.

For two classes $k$ and $r$,

$$
\log\frac{P(Y=k\mid\mathbf{x})}{P(Y=r\mid\mathbf{x})}
=\eta_k(\mathbf{x})-\eta_r(\mathbf{x}).
$$

The implementation stores one score equation for every class. Adding the same constant to all class scores does not alter the softmax probabilities; therefore, pairwise differences between class-score equations provide the conventional reference-class multinomial-logit representation.

---

## Project structure

Recommended layout:

```text
gam-project/
├── .venv/
├── data/
│   └── dataset.csv
├── outputs/
│   └── main-effects results
├── outputs_pairwise/
│   └── pairwise-interaction results
├── classical_gam_main.py
├── classical_gam_pairwise.py
├── comparison.py
├── inspect_final_gam.py
├── inspect_link_function.py
├── requirements.txt
└── README.md
```

The scripts assume they are executed from the project root. Run them after activating the same virtual environment used to fit and serialize the models.

---

## Installation

### 1. Create a virtual environment

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

Create `requirements.txt`:

```text
joblib
matplotlib
numpy
pandas
scikit-learn
```

Install:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Verify the environment

```powershell
python -c "import sys, sklearn, matplotlib, joblib; print(sys.version); print('sklearn', sklearn.__version__); print('matplotlib', matplotlib.__version__); print('joblib', joblib.__version__)"
```

---

## Running the analysis

### Main-effects GAM

```powershell
python classical_gam_main.py
```

Expected output directory:

```text
outputs/
```

### Pairwise-interaction GAM

```powershell
python classical_gam_pairwise.py
```

Expected output directory:

```text
outputs_pairwise/
```

### Compare the fitted model classes

Run the model comparison only after both nested-CV analyses have completed:

```powershell
python comparison.py
```

The script reads fold-level results from `outputs/` and `outputs_pairwise/`, aligns them by repeat and fold, and exports paired metric differences.

### Inspect final equations and coefficients

Main-effects model:

```powershell
python inspect_final_gam.py classic
```

Pairwise model:

```powershell
python inspect_final_gam.py pairwise
```

Use another reference class if required:

```powershell
python inspect_final_gam.py classic --reference-class B
python inspect_final_gam.py pairwise --reference-class B
```

Show more ranked transformed coefficients:

```powershell
python inspect_final_gam.py pairwise --top 30
```

### Inspect the link function numerically

Main-effects model:

```powershell
python inspect_link_function.py classic
```

Pairwise model:

```powershell
python inspect_link_function.py pairwise
```

The link-inspection utility should print raw class scores, model probabilities, manually calculated softmax probabilities, and their maximum numerical discrepancy.

### Syntax check before an expensive run

```powershell
python -m py_compile classical_gam_main.py
python -m py_compile classical_gam_pairwise.py
python -m py_compile comparison.py
python -m py_compile inspect_final_gam.py
python -m py_compile inspect_link_function.py
```

A successful compilation check returns to the prompt without output.

---

## Using comparison.py

`comparison.py` performs a paired comparison between the main-effects GAM and pairwise-interaction GAM using the outer-fold results saved by the training scripts.

### Prerequisites

Before running it, verify that these files exist:

```text
outputs/nested_cv_fold_results.csv
outputs_pairwise/nested_cv_fold_results.csv
```

The two analyses must use the same:

- dataset and row order;
- number of outer folds;
- number of repeats;
- outer random seed;
- stratification target.

A typical shared splitter is:

```python
RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42,
)
```

### Run

```powershell
python comparison.py
```

### What it calculates

The script merges results by `repeat` and `fold`, then calculates:

```text
log_loss_difference
accuracy_difference
balanced_accuracy_difference
macro_f1_difference
```

Differences use this direction:

$$
\Delta=\text{pairwise}-\text{main}.
$$

Interpret them as follows:

- negative log-loss difference favors the pairwise model;
- positive log-loss difference favors the main-effects model;
- positive accuracy, balanced-accuracy, or macro-F1 difference favors the pairwise model;
- negative accuracy, balanced-accuracy, or macro-F1 difference favors the main-effects model.

### Expected outputs

```text
outputs_pairwise/main_vs_pairwise_comparison.csv
outputs_pairwise/main_vs_pairwise_by_repeat.csv
```

`main_vs_pairwise_comparison.csv` contains one row per paired outer fold. `main_vs_pairwise_by_repeat.csv` averages the fold-level differences within each repeat, giving a more suitable descriptive view because the 25 fold results are not independent replications.

### Existing comparison result

The observed differences were:

| Metric | Mean difference | Standard deviation | Median difference |
|---|---:|---:|---:|
| Log loss | +0.023277 | 0.040034 | +0.039157 |
| Accuracy | -0.022000 | 0.031145 | -0.010000 |
| Balanced accuracy | -0.028990 | 0.021434 | -0.019757 |
| Macro-F1 | -0.028480 | 0.029338 | -0.011721 |

All four directions favor the main-effects model.

### Validation checks

The merge should use:

```python
validate="one_to_one"
```

A merge failure usually means that one file has duplicate or missing `repeat`/`fold` keys. Also compare the held-out `row_id` values when both analyses export nested-CV predictions. Matching fold labels alone is insufficient if one script has reordered or filtered rows.

### Common errors

**Missing file:** run both model scripts first and check the configured output directories.

**Merge creates no rows:** confirm that both files use the columns `repeat` and `fold` with matching values.

**Results appear reversed:** verify that differences are calculated as pairwise minus main, not main minus pairwise.

**Too many rows after merge:** use `validate="one_to_one"` and inspect duplicate fold identifiers.

---

## Using inspect_final_gam.py

`inspect_final_gam.py` exposes the final fitted class-score equations and all transformed coefficients for either saved model.

### Supported model types

```powershell
python inspect_final_gam.py classic
python inspect_final_gam.py pairwise
```

The utility expects:

```text
outputs/classical_gam_main_effects.joblib
outputs_pairwise/classical_gam_pairwise.joblib
```

### Why the two models need different transformer steps

The classical model uses a `ColumnTransformer` stored as:

```python
model.named_steps["preprocessor"]
```

The pairwise model uses its custom feature transformer stored as:

```python
model.named_steps["gam_features"]
```

Both pipelines use:

```python
model.named_steps["classifier"]
```

The inspection utility selects the correct transformer step automatically from the `classic` or `pairwise` command-line argument.

### Pairwise compatibility alias for existing models

An older pairwise model may have serialized the custom transformer as:

```text
__main__.PairwiseSplineFeatures
```

This occurs when `classical_gam_pairwise.py` defines the class locally and is executed directly. The immediate inspection utility imports that class and registers it before `joblib.load()`:

```python
import __main__

from classical_gam_pairwise import PairwiseSplineFeatures

__main__.PairwiseSplineFeatures = PairwiseSplineFeatures
```

Without this alias, loading may fail with:

```text
AttributeError: module '__main__' has no attribute 'PairwiseSplineFeatures'
```

The training file must remain importable and must protect execution with:

```python
if __name__ == "__main__":
    main()
```

For long-term model portability, move the custom transformer to a dedicated importable module and retrain the pairwise model. The compatibility alias is intended to inspect already fitted legacy files without retraining.

### Outputs for the classic model

```text
outputs/final_gam_components.csv
outputs/final_gam_equations.txt
outputs/final_gam_intercepts.csv
outputs/reference_class_link_equations.csv
```

### Outputs for the pairwise model

```text
outputs_pairwise/final_gam_components.csv
outputs_pairwise/final_gam_equations.txt
outputs_pairwise/final_gam_intercepts.csv
outputs_pairwise/reference_class_link_equations.csv
```

### Meaning of the files

`final_gam_components.csv` contains one row for each class and transformed model column. Typical fields are:

```text
class
component_type
original_component
component
coefficient
absolute_coefficient
```

`final_gam_equations.txt` contains exact transformed-space score equations such as:

```text
eta_B(x) = intercept
    + coefficient * main_spline__X1__basis_0
    - coefficient * main_spline__X1__basis_1
    ...
```

These are complete computational equations in the transformed feature space. A spline function is the weighted sum of all basis columns belonging to the same original predictor; an individual spline-basis coefficient should not be interpreted alone.

`final_gam_intercepts.csv` stores the class-specific intercepts.

`reference_class_link_equations.csv` subtracts the selected reference-class equation from every other class equation. With reference `O`, it provides transformed-space coefficients for:

```text
B versus O
G versus O
M versus O
```

### Reference-class option

Default:

```powershell
python inspect_final_gam.py classic --reference-class O
```

Alternative:

```powershell
python inspect_final_gam.py pairwise --reference-class B
```

Changing the reference class changes coefficient presentation but does not change predictions.

### Ranked coefficients

The `--top` option controls how many transformed coefficients are printed per class:

```powershell
python inspect_final_gam.py classic --top 20
```

Large transformed coefficients are not equivalent to globally important original predictors. Magnitudes depend on basis scaling, regularization, knot count, and feature distribution. Use effect curves, interaction surfaces, and held-out performance for scientific interpretation.

### Reserved-word implementation note

Because `class` is a Python keyword, do not use:

```python
row.class
```

when printing a DataFrame column named `class`. Use tuple unpacking:

```python
for class_name, intercept in intercepts.itertuples(
    index=False,
    name=None,
):
    print(
        f"eta_{class_name}: "
        f"intercept = {intercept:.10f}"
    )
```

### Common errors

**`AttributeError: module '__main__' has no attribute 'PairwiseSplineFeatures'`:** register the compatibility alias before loading the legacy pairwise model.

**`KeyError: 'preprocessor'`:** the pairwise model uses `gam_features`, not `preprocessor`.

**`SyntaxError` at `row.class`:** use tuple unpacking because `class` is reserved.

**Feature-name count differs from coefficient count:** verify that the same transformer class and compatible package versions are being used to load the model.

---

## Using inspect_link_function.py

`inspect_link_function.py` verifies the response/link structure used by the saved GAM pipeline and numerically checks the transformation from class scores to probabilities.

### Supported usage

```powershell
python inspect_link_function.py classic
python inspect_link_function.py pairwise
```

The utility should use the same model-selection and pairwise compatibility logic as `inspect_final_gam.py`.

### What it inspects

For the fitted classifier, it prints:

- estimator type;
- solver;
- class order;
- number of classes;
- coefficient-matrix shape;
- intercept-vector shape;
- raw class scores from `decision_function()`;
- model probabilities from `predict_proba()`;
- manually calculated softmax probabilities;
- maximum absolute difference between the two probability matrices.

### Mathematical check

For a score matrix $\eta$, stable softmax is calculated as:

```python
def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
```

The expected identity is:

$$
\operatorname{predict\_proba}(X)
\approx
\operatorname{softmax}(\operatorname{decision\_function}(X)).
$$

The maximum discrepancy should be close to floating-point precision, often around $10^{-15}$ or smaller.

### Input preparation

The raw validation DataFrame must have the same columns and representations used during training.

For the classic model, a typical order is:

```python
X = data[
    ["X1", "X2", "X4", "X5", "X7", "X6", "X3"]
].copy()
```

If training converted `X3` to strings, the inspection script must apply the same conversion:

```python
X["X3"] = X["X3"].astype(str)
```

For the pairwise model, pass a DataFrame because `PairwiseSplineFeatures` relies on original predictor names.

### Interpreting the results

`decision_function()` returns class-specific scores:

$$
\eta_B,\eta_G,\eta_M,\eta_O.
$$

`predict_proba()` converts them through softmax:

$$
p_k=\frac{e^{\eta_k}}{\sum_\ell e^{\eta_\ell}}.
$$

The scores are not probabilities and need not lie between zero and one. Only score differences affect the fitted probabilities. For classes $k$ and $r$:

$$
\log\frac{p_k}{p_r}=\eta_k-\eta_r.
$$

### Expected output

A successful run should report:

```text
Estimator type: LogisticRegression
Solver: lbfgs
Classes: ['B' 'G' 'M' 'O']
Number of classes: 4
Maximum difference between predict_proba and manual softmax: ...
```

The exact class order must be taken from `classifier.classes_`; do not assume the configured presentation order is the internal coefficient order.

### Common errors

**Model cannot be loaded:** apply the same pairwise compatibility alias used by `inspect_final_gam.py`.

**Feature names or columns do not match:** construct the input DataFrame with the original training columns and categorical representation.

**Probabilities do not match manual softmax:** ensure the comparison uses `model.decision_function(X)` from the complete pipeline rather than applying the classifier directly to raw predictors.

**Overflow in `exp`:** subtract the row maximum before exponentiation, as shown in the stable softmax implementation.

---

## Main-effects model

The main-effects script performs:

1. CSV loading and validation;
2. class-distribution audit;
3. unique-value and duplicate checks;
4. descriptive statistics and correlations;
5. repeated stratified nested cross-validation;
6. tuning of spline degree, knot count, and regularization strength;
7. final full-data descriptive fit;
8. export of fitted probabilities and coefficients;
9. class-specific main-effect plots.

### Hyperparameters

Typical configuration:

```python
n_knots_grid = (4, 5, 6, 7, 8)
degree_grid = (2, 3)
c_grid = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0)
```

Interpretation:

- `n_knots` controls spline basis flexibility;
- `degree` controls polynomial spline degree;
- `C` is inverse regularization strength;
- smaller `C` means stronger shrinkage;
- larger `C` means weaker shrinkage.

### Current main-effects result

The initial nested-CV result was approximately:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Multiclass log loss | 0.3294 | 0.0489 |
| Accuracy | 0.8696 | 0.0239 |
| Balanced accuracy | 0.8313 | 0.0325 |
| Macro-F1 | 0.8422 | 0.0305 |

Full-data metrics are descriptive only and must not be presented as generalization estimates.

---

## Pairwise-interaction model

The pairwise model retains every main effect and adds tensor-product interaction blocks for explicitly configured predictor pairs.

Default candidate pairs among smooth variables:

```python
candidate_pairs = (
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
```

### Interaction scaling

If an interaction design block is $Z$, the model may receive

$$
Z^*=\alpha Z,
$$

where `interaction_scale = α`.

Under L2 regularization, a smaller interaction scale requires larger coefficients to produce the same fitted surface and therefore creates stronger effective interaction shrinkage. `interaction_scale` is a tuning parameter, not a scientific effect measure.

### All-pairs result

The observed paired outer-fold differences were defined as

$$
\Delta = \text{pairwise GAM} - \text{main-effects GAM}.
$$

| Metric | Mean difference | Standard deviation | Median difference |
|---|---:|---:|---:|
| Log loss | +0.023277 | 0.040034 | +0.039157 |
| Accuracy | -0.022000 | 0.031145 | -0.010000 |
| Balanced accuracy | -0.028990 | 0.021434 | -0.019757 |
| Macro-F1 | -0.028480 | 0.029338 | -0.011721 |

Because lower log loss is better and higher classification metrics are better, all four comparisons favor the main-effects model. The all-pairs interaction model therefore should not be selected as the final model.

This result does not prove that every interaction is useless. It shows that including all ten pairs simultaneously reduces held-out performance.

---

## Choosing interaction pairs

Interaction selection must occur **inside the inner cross-validation loop**. Outer test folds must never be used to decide which pairs to retain.

### Recommended strategy: stability-aware forward selection

For each outer training set:

1. Fit and tune the main-effects model using only the outer training observations.
2. Test each candidate pair individually using inner CV.
3. Select the pair with the largest reduction in mean inner-CV log loss.
4. Add it only if the improvement exceeds a prespecified threshold.
5. Re-evaluate all remaining pairs conditionally on the selected pair.
6. Add at most one additional pair unless strong evidence supports more.
7. Fit the selected specification on the complete outer training set.
8. Evaluate it once on the untouched outer test fold.
9. Record selected pairs and their inner-CV gains.

### Recommended initial selection settings

```python
maximum_selected_pairs = 2
minimum_log_loss_improvement = 0.005
```

The improvement for a candidate pair is

$$
I=L_{\text{current}}-L_{\text{candidate}}.
$$

A positive value means the candidate reduces log loss.

### One-standard-error rule

Prefer the simpler model when its mean inner-CV log loss is within one standard error of the more complex candidate:

$$
L_{\text{simpler}}
\leq
L_{\text{best}}+SE(L_{\text{best}}).
$$

This prevents adding an interaction for an improvement too small to distinguish from resampling variation.

### Stability

For pair $p$, define selection stability as

$$
S_p=
\frac{\text{outer training sets selecting }p}
{\text{number of outer training sets}}.
$$

Pragmatic interpretation:

| Selection frequency | Interpretation |
|---:|---|
| Below 0.25 | Unstable evidence |
| 0.25–0.50 | Weak or fold-dependent evidence |
| 0.50–0.75 | Moderate evidence |
| Above 0.75 | Strong reproducibility |
| Above 0.90 | Very stable interaction |

A reasonable final retention rule is:

- selection frequency at least 0.60;
- mean inner-CV log-loss gain at least 0.005;
- no deterioration in mean outer-CV log loss;
- adequate two-dimensional data support;
- scientifically plausible interpretation.

Selecting zero interactions is a valid result.

### Do not select pairs using

- full-data coefficient magnitude alone;
- visually attractive surfaces;
- outer-test performance;
- the all-pairs fitted model alone;
- training accuracy;
- one isolated validation split.

Coefficient magnitudes depend on basis scaling, regularization, knot count, feature distribution, and predictor correlations. Held-out log-loss improvement and stability are more defensible selection criteria.

---

## Nested cross-validation

Nested CV separates hyperparameter and interaction selection from final performance estimation.

### Outer loop

The outer loop estimates generalization performance.

Typical configuration:

```python
outer_splits = 5
outer_repeats = 5
```

This produces 25 outer evaluations.

### Inner loop

The inner loop selects:

- spline knot count;
- spline degree;
- regularization strength `C`;
- interaction scale;
- interaction pairs, when pair selection is enabled.

Typical configuration:

```python
inner_splits = 5
```

### Important dependence note

The 25 fold-level results are not 25 independent experimental replications. Folds within a repeat share training observations, and observations reappear across repeated partitions. Report fold-level summaries descriptively and also average metric differences within each repeat.

---

## Model comparison

The main and pairwise scripts must use identical outer splits:

```python
RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42,
)
```

Both scripts must preserve the same row order.

### Comparison script

The repository utility is named `comparison.py`. It reads:

```text
outputs/nested_cv_fold_results.csv
outputs_pairwise/nested_cv_fold_results.csv
```

and merges them by `repeat` and `fold`.

The essential calculation is:

```python
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
```

Run:

```powershell
python comparison.py
```

### Difference direction

With

$$
\Delta=\text{pairwise}-\text{main},
$$

interpret differences as follows:

- log loss: negative favors pairwise;
- accuracy: positive favors pairwise;
- balanced accuracy: positive favors pairwise;
- macro-F1: positive favors pairwise.

---

## Interpreting outputs

### Nested-CV results

Use these for predictive conclusions:

```text
nested_cv_fold_results.csv
nested_cv_summary.csv
nested_cv_predictions.csv
nested_cv_confusion_matrix.csv
nested_cv_classification_report.csv
nested_cv_hyperparameter_frequencies.csv
```

### Full-data results

Use these for descriptive interpretation and plotting:

```text
full_data_descriptive_metrics.json
full_data_fitted_predictions.csv
final_model_coefficients.csv
main_effect_*.png
interaction_*_class_*.png
```

Full-data metrics are optimistic because the same observations were used for fitting and evaluation.

### Inspection outputs

`inspect_final_gam.py` adds exact transformed-space artifacts:

```text
final_gam_components.csv
final_gam_equations.txt
final_gam_intercepts.csv
reference_class_link_equations.csv
```

These are intended for auditing, reproducibility, and supplementary material. The `.joblib` pipeline remains the authoritative executable representation because it also stores fitted knots, standardization parameters, category levels, and transformation logic.

### Main-effect curves

A class-specific main-effect curve shows the centered additive score contribution

$$
f_{jk}(x_j).
$$

A positive value raises the score assigned to the corresponding class relative to the curve’s centering convention. It is not a probability change and must not be interpreted causally.

### Interaction surfaces

A class-specific interaction surface shows the tensor-product contribution

$$
f_{rsk}(x_r,x_s).
$$

Interpret surfaces only where observed data provide sufficient two-dimensional support. Correlated predictors may occupy a narrow diagonal region, leaving most of the rectangular plotting domain unsupported.

### Raw interaction-decomposition caution

The tensor-product interaction basis may overlap with main-effect-like variation unless interaction blocks are centered or orthogonalized relative to their component main effects. Predictions remain well defined, but a raw surface should be described cautiously as the fitted interaction block rather than automatically as a unique functional-ANOVA interaction.

---

## Known data considerations

### Class imbalance

Class `O` comprises 47.4% of the data. Therefore:

- ordinary accuracy may be optimistic relative to minority classes;
- balanced accuracy and macro-F1 must be reported;
- stratified CV is required;
- class-specific confusion matrices and recall should be examined.

### Correlated predictors

`X1`, `X2`, and `X4` appear strongly associated. This can cause:

- unstable attribution among main effects;
- redundant interaction pairs;
- poorly supported two-dimensional surfaces;
- fold-dependent pair selection.

An interaction between correlated variables may reproduce a shared latent regime rather than a genuine synergistic mechanism.

### Unusual X5 values

Most `X5` observations lie near 33, but 10 observations exceed 40, including regimes near 51, 98, and 178.

These values must not be silently deleted. Determine whether they represent:

- valid operating regimes;
- measurement errors;
- encoded states;
- unit-conversion problems;
- anomalies.

Any selected interaction involving `X5` should be subjected to a sensitivity analysis with the unusual regimes handled separately.

### Duplicates and grouping

There are two duplicate predictor groups with no conflicting labels. If observations belong to repeated specimens, instruments, batches, or runs, ordinary stratified CV should be replaced by grouped or blocked CV. Related observations must remain in the same fold.

---

## Windows and Matplotlib notes

The scripts generate image files but do not require GUI windows. Use Matplotlib’s non-interactive `Agg` backend before importing `matplotlib.pyplot`:

```python
import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
```

This prevents Tkinter errors such as:

```text
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
```

At the end of each Windows script, use:

```python
if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
```

Confirm the backend at startup:

```python
print("Matplotlib backend:", matplotlib.get_backend())
```

Expected output:

```text
Matplotlib backend: Agg
```

Run utilities from the ordinary VS Code integrated terminal rather than a Python Interactive or Jupyter window when checking serialization and subprocess behaviour.

---

## Computational cost

### `inner_n_jobs`

```python
inner_n_jobs = 1
```

uses one worker and is safest for debugging.

```python
inner_n_jobs = -1
```

uses all available logical CPU cores for the inner `GridSearchCV`.

Recommended setup:

```python
inner_n_jobs = -1
outer_n_jobs = 1
```

Only one CV level should usually be parallelized. Avoid setting both inner and outer jobs to `-1`, because nested parallelism can cause CPU oversubscription, high memory consumption, reduced performance, or an unresponsive computer.

A fixed worker count can offer a better interactive balance:

```python
inner_n_jobs = 4
```

### Main-effects grid size

For 5 knot values, 2 degrees, 6 values of `C`, 5 inner folds, and 25 outer evaluations:

$$
5\times2\times6\times5\times25=7500
$$

inner fits are required, excluding refits.

### Pairwise grid size

For 3 knot values, 2 degrees, 3 interaction scales, 5 values of `C`, 5 inner folds, and 25 outer evaluations:

$$
3\times2\times3\times5\times5\times25=11250
$$

inner fits are required, excluding refits.

### Quick debugging configuration

Before a full research run, use:

```python
outer_splits = 5
outer_repeats = 1
inner_splits = 3

n_knots_grid = (3, 4)
degree_grid = (2,)
c_grid = (0.1, 1.0, 10.0)
interaction_scale_grid = (0.5, 1.0)
inner_n_jobs = 1
```

After successful completion, restore the full configuration.

The inspection and comparison utilities do not refit the models and should run much faster than the training scripts.

---

## Reproducibility

For reproducible comparisons:

1. preserve the original row order;
2. use the same outer splitter and random seed;
3. save every selected hyperparameter;
4. save held-out row IDs and probabilities;
5. record package versions;
6. keep model-selection logic inside inner CV;
7. save the final configuration with every run;
8. do not overwrite previous output directories without archiving them;
9. retain `comparison.py`, `inspect_final_gam.py`, and `inspect_link_function.py` with the archived model;
10. retain the source defining every custom transformer used by a serialized model.

Recommended environment export:

```powershell
python -m pip freeze > environment-lock.txt
```

Recommended version information to save:

```python
import platform
import sys

import joblib
import matplotlib
import numpy
import pandas
import sklearn

information = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit_learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
    "joblib": joblib.__version__,
}
```

### Serialization note

A `.joblib` file containing custom Python objects depends on the original module path and class name. Archive these together:

```text
classical_gam_pairwise.joblib
classical_gam_pairwise.py or gam_transformers.py
environment-lock.txt
analysis_configuration.json
final_gam_components.csv
final_gam_intercepts.csv
reference_class_link_equations.csv
```

Text-based exports remain inspectable even if future package changes make the executable serialized model difficult to load.

---

## Limitations

1. **This is predictive, not causal.** Main-effect and interaction functions describe conditional associations.
2. **The implementation uses L2 coefficient regularization.** It is not identical to a statistical GAM using an explicit integrated curvature penalty.
3. **Interaction scaling and `C` are partially coupled.** Their selected numerical values are tuning results, not scientific quantities.
4. **Correlated predictors complicate attribution.** Different folds may distribute signal among `X1`, `X2`, and `X4` differently.
5. **Rare regimes may dominate interactions.** This is particularly relevant to `X5`.
6. **Repeated CV fold results are dependent.** Treat simple confidence intervals and fold-level tests cautiously.
7. **Strong class separation may be engineered.** Dataset provenance is necessary to determine whether `Y` was assigned using thresholds related to the predictors.
8. **Interaction surfaces outside observed support are extrapolations.** They should not be scientifically interpreted.
9. **Transformed coefficient magnitude is not original-variable importance.** Spline coefficients must be interpreted jointly through the fitted functions.
10. **Legacy pairwise model files may depend on `__main__.PairwiseSplineFeatures`.** Use the inspection compatibility alias or retrain with the class in an importable module.

---

## Recommended research workflow

### Stage 1 — Main-effects GAM

- audit the data;
- fit the main-effects model;
- estimate performance with repeated nested CV;
- inspect class-specific errors;
- inspect smooth functions and correlated predictors;
- run `inspect_final_gam.py classic`;
- run `inspect_link_function.py classic`.

### Stage 2 — All-pairs GAM

- fit all ten smooth-variable interaction pairs;
- compare with the main-effects model using identical outer folds;
- treat this as a complexity stress test;
- run `inspect_final_gam.py pairwise`;
- run `inspect_link_function.py pairwise`.

Current result: the all-pairs model performs worse than the main-effects model.

### Stage 3 — Paired comparison

- run `comparison.py` only after both nested-CV runs finish;
- inspect fold-level paired differences;
- inspect differences averaged within repeats;
- verify that held-out row IDs match between model families;
- do not treat the 25 outer folds as 25 independent experiments.

### Stage 4 — Selected-pair GAM

- perform stability-aware forward selection inside inner CV;
- allow at most two pairs initially;
- apply a minimum log-loss gain and one-standard-error rule;
- record pair-selection frequencies;
- evaluate selected specifications on outer folds only once.

### Stage 5 — Sensitivity analyses

- examine unusual `X5` regimes;
- assess reduction of the correlated `X1`/`X2`/`X4` block;
- compare maximum one versus two interactions;
- inspect whether selected surfaces are supported by observed data;
- compare linear, categorical, and excluded representations of low-resolution predictors when scientifically justified;
- use grouped CV if repeated measurements or batches exist.

### Stage 6 — Final reporting

Report:

- nested-CV log loss, accuracy, balanced accuracy, and macro-F1;
- per-class recall, precision, and F1;
- confusion matrices;
- paired main-versus-interaction differences;
- pair-selection frequencies;
- mean conditional inner-CV gain;
- sensitivity-analysis results;
- class-specific main-effect and supported interaction plots;
- fitted link structure verified by `inspect_link_function.py`;
- transformed equations and reference-class contrasts exported by `inspect_final_gam.py`.

A suitable final conclusion may be that the main-effects GAM is preferred. Selecting no interaction is a valid and potentially important scientific result.

---

## Current conclusion

The main-effects model currently provides the stronger balance of predictive quality, parsimony, and interpretability. Adding all ten pairwise tensor-product interactions increased mean held-out log loss and reduced accuracy, balanced accuracy, and macro-F1.

The next defensible step is not to search the outer results for a favorable pair. It is to perform pair selection entirely inside the inner CV loop, quantify selection stability across outer resamples, and retain only interactions that provide reproducible held-out benefit and have adequate data support.

For reproducible auditing, use:

```powershell
python comparison.py
python inspect_final_gam.py classic
python inspect_final_gam.py pairwise
python inspect_link_function.py classic
python inspect_link_function.py pairwise
```

Together, these utilities separate three distinct tasks: `comparison.py` evaluates paired held-out performance, `inspect_final_gam.py` exposes the fitted score equations and coefficients, and `inspect_link_function.py` verifies how those scores are converted into multinomial probabilities.
