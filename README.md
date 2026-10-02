# nannyrust

A from-scratch Rust reimplementation of NannyML's core statistical metrics - algorithms rewritten in Rust.

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

## Known gaps 

1. **CBPE requires calibrated probabilities.** NannyML calibrates
   `y_pred_proba` against the reference set (typically via isotonic
   regression) before running CBPE. This crate does not implement a
   calibrator — feed it already-calibrated probabilities, or add one
   (e.g. pool-adjacent-violators for isotonic regression) upstream.
2. **DLE needs an auxiliary regression model.** NannyML trains a LightGBM
   model on the reference set to predict each row's error. Training that
   model is a full ML pipeline, out of scope here — implement the
   `LossEstimator` trait over whatever regressor you bring (`linfa`,
   `smartcore`, an FFI call to a trained LightGBM booster, etc.). This
   crate only implements the final reduction step (turning per-row
   predicted losses into MAE/MSE/RMSE estimates).
4. Continuous binning (`binning::reference_bin_edges`) approximates
   NannyML's Doane's-formula + `numpy.histogram` binning; edge behavior at
   bin boundaries may differ by a bin in rare cases — worth a
   side-by-side check against the Python output on your actual data
   before switching over.

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
