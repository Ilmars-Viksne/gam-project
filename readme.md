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

\[
\mathbf{x}=(X_1,X_2,X_3,X_4,X_5,X_6,X_7),
\]

and let the response contain four unordered classes:

\[
Y\in\{O,B,M,G\}.
\]

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

For class \(k\), the class-specific score is

\[
\eta_k(\mathbf{x})=
\beta_{0k}
+f_{1k}(X_1)
+f_{2k}(X_2)
+f_{4k}(X_4)
+f_{5k}(X_5)
+f_{7k}(X_7)
+\beta_{6k}\widetilde{X}_6
+\gamma_k(X_3),
\]

where:

- each \(f_{jk}\) is a univariate B-spline expansion;
- \(\widetilde{X}_6\) is standardized `X6`;
- \(\gamma_k(X_3)\) is a categorical main effect;
- no predictor products or pairwise surfaces are included.

Class probabilities are produced by the softmax transformation:

\[
P(Y=k\mid\mathbf{x})=
\frac{\exp(\eta_k(\mathbf{x}))}
{\sum_\ell \exp(\eta_\ell(\mathbf{x}))}.
\]

The implementation is most precisely described as a **penalized multinomial logistic additive B-spline model**. L2 regularization is applied to the resulting spline and main-effect coefficients.

### Multiclass GAM with pairwise interactions

The interaction model extends the class score to

\[
\eta_k(\mathbf{x})=
\beta_{0k}
+\sum_j f_{jk}(x_j)
+\sum_{(r,s)\in\mathcal I} f_{rsk}(x_r,x_s).
\]

For a selected pair \((X_r,X_s)\), the interaction is represented using a tensor-product spline basis:

\[
f_{rsk}(x_r,x_s)=
\sum_a\sum_b
\theta_{krsab} B_{ra}(x_r)B_{sb}(x_s).
\]

All component main effects remain in the model. This follows the hierarchy principle: an interaction is not fitted without its corresponding main effects.

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
├── compare_models.py
├── requirements.txt
└── readme.md
```

The scripts assume they are executed from the project root.

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

### Syntax check before an expensive run

```powershell
python -m py_compile classical_gam_main.py
python -m py_compile classical_gam_pairwise.py
```

A successful compilation check returns to the prompt without output.

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

If an interaction design block is \(Z\), the model may receive

\[
Z^*=\alpha Z,
\]

where `interaction_scale = α`.

Under L2 regularization, a smaller interaction scale requires larger coefficients to produce the same fitted surface and therefore creates stronger effective interaction shrinkage. `interaction_scale` is a tuning parameter, not a scientific effect measure.

### All-pairs result

The observed paired outer-fold differences were defined as

\[
\Delta = \text{pairwise GAM} - \text{main-effects GAM}.
\]

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

\[
I=L_{\text{current}}-L_{\text{candidate}}.
\]

A positive value means the candidate reduces log loss.

### One-standard-error rule

Prefer the simpler model when its mean inner-CV log loss is within one standard error of the more complex candidate:

\[
L_{\text{simpler}}
\leq
L_{\text{best}}+SE(L_{\text{best}}).
\]

This prevents adding an interaction for an improvement too small to distinguish from resampling variation.

### Stability

For pair \(p\), define selection stability as

\[
S_p=
\frac{\text{outer training sets selecting }p}
{\text{number of outer training sets}}.
\]

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

Create `compare_models.py`:

```python
from pathlib import Path

import pandas as pd


main_results = pd.read_csv(
    Path("outputs") / "nested_cv_fold_results.csv"
)

pairwise_results = pd.read_csv(
    Path("outputs_pairwise") / "nested_cv_fold_results.csv"
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

difference_columns = [
    "log_loss_difference",
    "accuracy_difference",
    "balanced_accuracy_difference",
    "macro_f1_difference",
]

print(
    comparison[difference_columns]
    .agg(["mean", "std", "median"])
    .T
)

repeat_summary = (
    comparison
    .groupby("repeat")[difference_columns]
    .mean()
)

print("\nDifferences averaged within repeats")
print(repeat_summary)

comparison.to_csv(
    Path("outputs_pairwise")
    / "main_vs_pairwise_comparison.csv",
    index=False,
)

repeat_summary.to_csv(
    Path("outputs_pairwise")
    / "main_vs_pairwise_by_repeat.csv"
)
```

Run:

```powershell
python compare_models.py
```

### Difference direction

With

\[
\Delta=\text{pairwise}-\text{main},
\]

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

### Main-effect curves

A class-specific main-effect curve shows the centered additive score contribution

\[
f_{jk}(x_j).
\]

A positive value raises the score assigned to the corresponding class relative to the curve’s centering convention. It is not a probability change and must not be interpreted causally.

### Interaction surfaces

A class-specific interaction surface shows the pure tensor-product contribution

\[
f_{rsk}(x_r,x_s).
\]

Interpret surfaces only where observed data provide sufficient two-dimensional support. Correlated predictors may occupy a narrow diagonal region, leaving most of the rectangular plotting domain unsupported.

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

\[
5\times2\times6\times5\times25=7500
\]

inner fits are required, excluding refits.

### Pairwise grid size

For 3 knot values, 2 degrees, 3 interaction scales, 5 values of `C`, 5 inner folds, and 25 outer evaluations:

\[
3\times2\times3\times5\times5\times25=11250
\]

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
8. do not overwrite previous output directories without archiving them.

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

---

## Recommended research workflow

### Stage 1 — Main-effects GAM

- audit the data;
- fit the main-effects model;
- estimate performance with repeated nested CV;
- inspect class-specific errors;
- inspect smooth functions and correlated predictors.

### Stage 2 — All-pairs GAM

- fit all ten smooth-variable interaction pairs;
- compare with the main-effects model using identical outer folds;
- treat this as a complexity stress test.

Current result: the all-pairs model performs worse than the main-effects model.

### Stage 3 — Selected-pair GAM

- perform stability-aware forward selection inside inner CV;
- allow at most two pairs initially;
- apply a minimum log-loss gain and one-standard-error rule;
- record pair-selection frequencies;
- evaluate selected specifications on outer folds only once.

### Stage 4 — Sensitivity analyses

- examine unusual `X5` regimes;
- assess reduction of the correlated `X1`/`X2`/`X4` block;
- compare maximum one versus two interactions;
- inspect whether selected surfaces are supported by observed data;
- use grouped CV if repeated measurements or batches exist.

### Stage 5 — Final reporting

Report:

- nested-CV log loss, accuracy, balanced accuracy, and macro-F1;
- per-class recall, precision, and F1;
- confusion matrices;
- paired main-versus-interaction differences;
- pair-selection frequencies;
- mean conditional inner-CV gain;
- sensitivity-analysis results;
- class-specific main-effect and supported interaction plots.

A suitable final conclusion may be that the main-effects GAM is preferred. Selecting no interaction is a valid and potentially important scientific result.

---

## Current conclusion

The main-effects model currently provides the stronger balance of predictive quality, parsimony, and interpretability. Adding all ten pairwise tensor-product interactions increased mean held-out log loss and reduced accuracy, balanced accuracy, and macro-F1.

The next defensible step is not to search the outer results for a favorable pair. It is to perform pair selection entirely inside the inner CV loop, quantify selection stability across outer resamples, and retain only interactions that provide reproducible held-out benefit and have adequate data support.
