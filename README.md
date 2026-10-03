# nannyrust

A from-scratch Rust reimplementation of NannyML's core statistical metrics: univariate drift detection, realised classification metrics and label-free performance estimation. Rust standard library only, no third-party dependencies.

**Status:** v0.1.0. All 8 unit tests pass and the library has been benchmarked against SciPy, scikit-learn and NannyML (tables below). It is **not** a drop-in numerical replacement for NannyML yet: see [Differences from NannyML](#differences-from-nannyml).

## What's included

| NannyML method name | Rust function |
|---|---|
| `kolmogorov_smirnov` | `continuous::kolmogorov_smirnov` |
| `wasserstein` | `continuous::wasserstein` |
| `jensen_shannon` (continuous) | `continuous::jensen_shannon` |
| `hellinger` (continuous) | `continuous::hellinger` |
| `chi2` | `categorical::chi2` |
| `l_infinity` | `categorical::l_infinity` |
| `jensen_shannon` (categorical) | `categorical::jensen_shannon` |
| `hellinger` (categorical) | `categorical::hellinger` |
| Realized accuracy/precision/recall/F1/specificity | `performance::ConfusionMatrix` |
| ROC AUC | `performance::roc_auc` |
| CBPE (Confidence-based Performance Estimation) | `performance::cbpe_estimate` |
| DLE (Direct Loss Estimation) | `performance::{LossEstimator, estimated_mae, estimated_mse, estimated_rmse}` |

## Usage

```rust
use nannyml_metrics::continuous;
use nannyml_metrics::categorical;
use nannyml_metrics::performance;

let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let analysis = vec![3.0, 4.0, 5.0, 6.0, 7.0];

let ks = continuous::kolmogorov_smirnov(&reference, &analysis);
let js = continuous::jensen_shannon(&reference, &analysis);

let y_pred = vec![true, false, true];
let y_pred_proba = vec![0.9, 0.2, 0.7]; // must already be calibrated
let cm = performance::cbpe_estimate(&y_pred, &y_pred_proba);
println!("estimated accuracy: {}", cm.accuracy());
```

## Benchmarks

### Test setup

| Item | Value |
|---|---|
| Rust | 1.75.0 (the shipped `Cargo.lock` is format v4 and needs cargo ≥ 1.78; it was regenerated for this run) |
| Python / NumPy / SciPy | 3.12.3 / 2.4.4 / 1.17.1 |
| scikit-learn / NannyML | 1.8.0 / 0.13.1 |
| Hardware | 1 vCPU container, single-threaded |
| NannyML usage | method classes called directly (`_fit` on reference, `_calculate` on analysis), bypassing chunking and thresholds |

All data are simulated except the case study (E5). "Max abs diff" is the maximum absolute difference between nannyrust v0.1.0 and the stated reference over all replicates.

### E1: Continuous features (9 scenarios × 20 replicates = 180 datasets)

Scenarios: S1 null, N(5, 1.5²), n = 5,000 · S2 batch mean shift +0.5 · S3 variance inflation (SD 1.5 → 2.5) · S4 unimodal → bimodal · S5 lognormal coverage (+20% scale) · S6 negative-binomial integer read counts (mean 30 → 36) · S7 small n (200 vs 200) · S8 analysis range exceeds reference range · S9 large n (50,000 vs 50,000, shift 0.1).

**KS and Wasserstein (exact):**

| Scenario | KS vs SciPy | Wasserstein vs SciPy | KS vs NannyML | Wasserstein vs NannyML |
|---|---|---|---|---|
| S1 | 6.9e-17 | 4.6e-15 | 6.9e-17 | 4.6e-15 |
| S2 | 8.3e-17 | 4.9e-14 | 8.3e-17 | 4.9e-14 |
| S3 | 8.3e-17 | 4.8e-14 | 8.3e-17 | 4.8e-14 |
| S4 | 8.3e-17 | 4.6e-14 | 8.3e-17 | 4.6e-14 |
| S5 | 5.6e-17 | 5.0e-13 | 5.6e-17 | 5.0e-13 |
| S6 | 8.3e-17 | 1.8e-15 | 8.3e-17 | 1.8e-15 |
| S7 | 5.6e-17 | 5.0e-14 | 5.6e-17 | 5.0e-14 |
| S8 | 1.1e-16 | 5.0e-14 | 1.1e-16 | 5.0e-14 |
| S9 | 1.0e-16 | 4.5e-14 | **1.2e-4** | **1.8e-4** |

nannyrust is exact. NannyML is exact below 10,000 reference rows and switches to a histogram approximation at or above that size, which explains the S9 difference.

**Jensen-Shannon and Hellinger (v0.1.0 vs NannyML):**

| Scenario | JS ratio (Rust/NannyML, mean) | Max JS diff (raw) | Max JS diff after ÷ √ln 2 | Max Hellinger diff | Median Hellinger diff | Replicates with same bin count |
|---|---|---|---|---|---|---|
| S1 | 0.772 | 0.0135 | 0.0088 | 0.0088 | 0.0039 | 45% |
| S2 | 0.825 | 0.0308 | 0.0061 | 0.0056 | 0.0009 | 40% |
| S3 | 0.824 | 0.0538 | 0.0076 | 0.0128 | 0.0079 | 60% |
| S4 | 0.831 | 0.0792 | 0.0125 | 0.0117 | 4.0e-14 | 75% |
| S5 | 0.825 | 0.0309 | 0.0081 | 0.0081 | 0.0022 | 0% |
| S6 | 0.829 | 0.0407 | 0.0044 | 0.0056 | 0.0009 | 80% |
| S7 (n = 200) | 0.756 | 0.1010 | 0.0665 | 0.0683 | 0.0163 | 40% |
| S8 | 0.829 | 0.0663 | 0.0037 | 0.0134 | 0.0081 | 50% |
| S9 | 0.830 | 0.0069 | 0.0022 | 0.0018 | 0.0004 | 50% |

The JS ratio tends to √ln 2 = 0.8326 when drift is appreciable (v0.1.0 uses natural logs; NannyML uses base 2). Remaining differences come from histogram binning (see below).

**Compatibility variant** (NumPy `doane` bin edges, `np.histogram` semantics, base-2 JS; evaluation code in the supplement, not part of v0.1.0), same 180 datasets:

| Metric vs NannyML | Max abs diff |
|---|---|
| Jensen-Shannon | 4.9e-14 |
| Hellinger | 4.9e-14 |

### E2: Categorical features (6 scenarios × 20 replicates = 120 datasets)

Scenarios: C1 null (4 sequencing platforms, n = 3,000) · C2 proportion shift · C3 unseen category appears · C4 binary feature (df = 1) · C5 high cardinality (200 categories, n = 5,000) · C6 small n (150 vs 150).

Degrees of freedom matched SciPy in 120/120 comparisons.

| Scenario | χ² stat vs SciPy (no correction) | χ² p-value vs SciPy | χ² stat vs SciPy (Yates, NannyML default) | L-inf vs NannyML | JS ratio | JS diff after ÷ √ln 2 | Hellinger vs NannyML |
|---|---|---|---|---|---|---|---|
| C1 | 4.9e-13 | 4.8e-14 | 4.9e-13 | 3.4e-15 | 0.833 | 5.9e-15 | 4.8e-15 |
| C2 | 4.8e-11 | 7.3e-48 | 4.8e-11 | 3.3e-14 | 0.833 | 5.7e-14 | 4.9e-14 |
| C3 | 4.8e-11 | 5.4e-148 | 4.8e-11 | 3.3e-14 | 0.833 | 6.0e-14 | 4.8e-14 |
| C4 (df = 1) | 4.9e-12 | 3.6e-17 | **0.322** | 3.4e-15 | 0.833 | 6.2e-15 | 4.8e-15 |
| C5 | 3.2e-10 | 8.4e-85 | 3.2e-10 | 3.5e-18 | 0.833 | 5.7e-14 | 4.7e-14 |
| C6 | 4.3e-12 | 2.4e-14 | 4.3e-12 | 3.3e-14 | 0.833 | 6.0e-14 | 4.6e-14 |

For a 2 × 2 table (df = 1), SciPy and NannyML apply Yates' continuity correction by default; nannyrust does not. In C4 the p-value differs by up to 1.2e-4.

### E3: Classification metrics and CBPE (4 scenarios × 100 replicates, n = 2,000)

**Realised metrics and ROC AUC vs scikit-learn** (400 datasets, 30% with deliberately tied scores):

| Metric | Max abs diff |
|---|---|
| Accuracy | 0 |
| Precision | 4.9e-14 |
| Recall | 5.0e-14 |
| Specificity | 5.0e-14 |
| F1 | 5.0e-14 |
| ROC AUC | 5.0e-14 |

**CBPE estimate minus realised value** (bias = mean error; MAE = mean absolute error):

| Scenario | Acc. bias | Prec. bias | Recall bias | Spec. bias | F1 bias | Acc. MAE | Prec. MAE | Recall MAE | Spec. MAE | F1 MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 calibrated, no shift | +0.0009 | +0.0013 | +0.0010 | +0.0008 | +0.0012 | 0.0071 | 0.0119 | 0.0091 | 0.0069 | 0.0090 |
| P2 covariate shift, calibration holds | +0.0003 | −0.0006 | +0.0008 | −0.0003 | +0.0001 | 0.0076 | 0.0081 | 0.0063 | 0.0109 | 0.0060 |
| P3 overconfident model | +0.1000 | +0.0937 | +0.1230 | +0.0746 | +0.1102 | 0.1000 | 0.0937 | 0.1230 | 0.0746 | 0.1102 |
| P4 underconfident model | −0.1028 | −0.0975 | −0.1265 | −0.0768 | −0.1138 | 0.1028 | 0.0975 | 0.1265 | 0.0768 | 0.1138 |

CBPE is unbiased when probabilities are calibrated and biased by about ±10 percentage points when they are not. **The crate does not calibrate probabilities.**

### E4: Runtime (one core, median of 3–5 runs, release build)

Rust timings include everything done per call (sorting or binning the reference). NannyML timings are `_calculate` only, after `_fit`, so reference preparation is excluded, and NannyML uses its histogram approximation for KS and Wasserstein when the reference has ≥ 10,000 rows. SciPy timings are exact.

| n per sample | Rust KS | Rust Wasserstein | Rust JS | Rust Hellinger |
|---|---|---|---|---|
| 1,000 | 0.26 ms | 0.23 ms | 0.08 ms | 0.06 ms |
| 10,000 | 3.09 ms | 3.08 ms | 1.00 ms | 0.97 ms |
| 100,000 | 34.8 ms | 34.1 ms | 12.0 ms | 13.1 ms |
| 1,000,000 | 0.396 s | 0.397 s | 0.143 s | 0.135 s |
| 10,000,000 | 4.59 s | 4.50 s | 1.67 s | 1.69 s |

| n per sample | SciPy KS (exact) | SciPy Wasserstein (exact) | NannyML KS | NannyML Wasserstein | NannyML JS | NannyML Hellinger |
|---|---|---|---|---|---|---|
| 1,000 | 0.93 ms | 0.25 ms | 1.25 ms | 0.39 ms | 0.20 ms | 0.17 ms |
| 10,000 | 3.98 ms | 2.96 ms | 0.76 ms | 0.74 ms | 0.22 ms | 0.17 ms |
| 100,000 | 63.8 ms | 43.8 ms | 2.20 ms | 2.93 ms | 1.57 ms | 1.26 ms |
| 1,000,000 | 0.325 s | 0.506 s | 0.015 s | 0.016 s | 0.008 s | 0.008 s |
| 10,000,000 | 3.86 s | 7.07 s | 0.152 s | 0.144 s | 0.077 s | 0.078 s |

At n = 10⁷, nannyrust is about 1.2× slower than exact SciPy for KS, 1.6× faster for Wasserstein, and roughly 30× (KS, Wasserstein) and 22× (JS, Hellinger) slower than NannyML's approximate path. Peak memory was not measured.

### E5: Case study, Wisconsin diagnostic breast cancer data

569 samples, 30 features, malignant = positive. 100 random splits (190 train / 190 calibration-reference / 189 analysis). Logistic regression calibrated by isotonic regression in Python (the crate has no calibrator). The nine size features (radius, perimeter, area; mean, SE, worst) were multiplied by 1.15 or 1.40 to imitate a measurement batch effect. A feature is flagged when KS > 0.1395 (asymptotic α = 0.05). Means over splits:

| Condition | Realised accuracy | CBPE accuracy | Bias | MAE | Mean AUC | Features flagged (of 30) | Size features flagged (of 9) | Other flagged (of 21) | Mean KS, size | Mean KS, other |
|---|---|---|---|---|---|---|---|---|---|---|
| No change | 0.970 | 0.977 | +0.007 | 0.013 | 0.986 | 1.4 | 0.4 | 1.0 | 0.086 | 0.087 |
| Size features ×1.15 | 0.937 | 0.968 | +0.031 | 0.033 | 0.971 | 8.5 | 7.5 | 1.0 | 0.214 | 0.087 |
| Size features ×1.40 | 0.807 | 0.956 | +0.149 | 0.149 | 0.886 | 10.0 | 9.0 | 1.0 | 0.415 | 0.087 |

Drift detection found the affected features, but CBPE largely missed the accuracy loss because scaling changes the relationship between observed features and label, which a reference-set calibration cannot represent. Treat feature-level drift as the primary signal in such cases.

### Reproducing the benchmarks

A harness (`examples/harness.rs`) reads binary arrays, calls the library and prints results and timings. It, the Python experiment scripts and the raw per-replicate CSVs are in `nannyrust_paper_supplement.zip`.

```bash
cp rust_harness/harness.rs examples/
cargo build --release --example harness
python scripts/e1_continuous.py   # E1
python scripts/e2_cat.py          # E2
python scripts/e3_perf.py         # E3
python scripts/e4_scale.py        # E4
python scripts/e5_case.py         # E5
```

The scripts assume the build at `/tmp/nannyrust`; edit the `H=` path at the top of each script. Requires `scipy`, `scikit-learn`, `pandas` and `nannyml`.

## Differences from NannyML

Measured above, relevant if you compare outputs or reuse thresholds tuned on NannyML:

1. **Jensen-Shannon uses natural logarithms** (maximum √ln 2 ≈ 0.83); NannyML uses base 2 (maximum 1). Values are lower by a factor of 0.83.
2. **Histogram binning differs from NumPy/NannyML** for JS and Hellinger: the Doane bin count is rounded (NumPy: ceiling), a distinct-value rule is applied (NumPy: none), and out-of-range analysis values are clamped into edge bins (NannyML: dropped, remaining mass placed in an extra bin). Differences reach about 0.01 at n ≈ 5,000 and 0.07 at n = 200.
3. **No Yates correction** for 2 × 2 chi-squared tables; NannyML/SciPy apply it by default.
4. **Exact KS and Wasserstein** at all sizes; NannyML approximates from 10,000 reference rows.
