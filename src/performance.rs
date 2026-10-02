//! Performance metrics and the estimators NannyML builds on top of them:
//! - CBPE (Confidence-based Performance Estimation) for classification
//! - DLE (Direct Loss Estimation) for regression
//!
//! Ground-truth ("realized") metrics are provided too, since both
//! estimators are validated against them once labels arrive.

// ---------------------------------------------------------------------
// Realized classification metrics (labels available)
// ---------------------------------------------------------------------

/*
Gaurav Sablok
gsablok@proton.me
 */

#[derive(Debug, Clone, Copy, Default)]
pub struct ConfusionMatrix {
    pub tp: f64,
    pub fp: f64,
    pub fn_: f64,
    pub tn: f64,
}

impl ConfusionMatrix {
    pub fn from_labels(y_true: &[bool], y_pred: &[bool]) -> Self {
        let mut cm = ConfusionMatrix::default();
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            match (t, p) {
                (true, true) => cm.tp += 1.0,
                (false, true) => cm.fp += 1.0,
                (true, false) => cm.fn_ += 1.0,
                (false, false) => cm.tn += 1.0,
            }
        }
        cm
    }

    pub fn accuracy(&self) -> f64 {
        let n = self.tp + self.fp + self.fn_ + self.tn;
        if n == 0.0 {
            0.0
        } else {
            (self.tp + self.tn) / n
        }
    }

    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0.0 {
            0.0
        } else {
            self.tp / denom
        }
    }

    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0.0 {
            0.0
        } else {
            self.tp / denom
        }
    }

    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0.0 {
            0.0
        } else {
            self.tn / denom
        }
    }

    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// ROC AUC via the Mann-Whitney U statistic (rank-based, no thresholding
/// needed, handles ties by average-ranking).
pub fn roc_auc(y_true: &[bool], y_score: &[f64]) -> f64 {
    let n = y_true.len();
    assert_eq!(n, y_score.len());

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| y_score[a].partial_cmp(&y_score[b]).unwrap());

    // Average ranks (1-indexed) to correctly handle tied scores.
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && y_score[idx[j + 1]] == y_score[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in i..=j {
            ranks[idx[k]] = avg_rank;
        }
        i = j + 1;
    }

    let n_pos = y_true.iter().filter(|&&t| t).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        // Undefined when only one class is present. scikit-learn raises in
        // this case; we return NaN so the caller can decide how to handle it.
        return f64::NAN;
    }

    let sum_ranks_pos: f64 = ranks
        .iter()
        .zip(y_true.iter())
        .filter(|(_, &t)| t)
        .map(|(&r, _)| r)
        .sum();

    let u_pos = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0;
    u_pos / (n_pos * n_neg)
}

// ---------------------------------------------------------------------
// CBPE: Confidence-based Performance Estimation (classification, no labels)
// ---------------------------------------------------------------------
//
// CBPE assumes `y_pred_proba` is well-calibrated (NannyML calibrates it
// against the reference set before this step — do the same upstream).
// Each row then contributes probabilistically to the confusion matrix
// instead of deterministically, since the true label is unknown:
//
//   predicted positive row -> P(y=1) mass to TP, P(y=0) mass to FP
//   predicted negative row -> P(y=1) mass to FN, P(y=0) mass to TN

pub fn cbpe_confusion_matrix(y_pred: &[bool], y_pred_proba: &[f64]) -> ConfusionMatrix {
    let mut cm = ConfusionMatrix::default();
    for (&pred, &proba) in y_pred.iter().zip(y_pred_proba.iter()) {
        if pred {
            cm.tp += proba;
            cm.fp += 1.0 - proba;
        } else {
            cm.fn_ += proba;
            cm.tn += 1.0 - proba;
        }
    }
    cm
}

/// Convenience wrapper: estimate accuracy/precision/recall/F1/specificity
/// directly from predictions + calibrated probabilities, no labels needed.
pub fn cbpe_estimate(y_pred: &[bool], y_pred_proba: &[f64]) -> ConfusionMatrix {
    cbpe_confusion_matrix(y_pred, y_pred_proba)
}

// ---------------------------------------------------------------------
// DLE: Direct Loss Estimation (regression, no labels)
// ---------------------------------------------------------------------
//
// DLE trains an auxiliary regressor (in NannyML: LightGBM) on the
// reference set to predict the primary model's per-row absolute/squared
// error from the input features. That training step is a full ML model
// and is out of scope for a metrics crate — bring your own regressor
// (e.g. `linfa`, `smartcore`, or an FFI call to a trained LightGBM model)
// and implement `LossEstimator`. What we *do* implement is the reduction
// step: turning per-row predicted losses into the final estimated metric,
// which is what NannyML's DLE does once the auxiliary model has scored
// each row.

/// Implement this over your trained auxiliary loss-predicting model.
pub trait LossEstimator {
    /// Predicted absolute error |y_true - y_pred| for each row, estimated
    /// from features alone (no ground truth required at inference time).
    fn predict_absolute_error(&self, features: &[Vec<f64>]) -> Vec<f64>;

    /// Predicted squared error (y_true - y_pred)^2 for each row.
    fn predict_squared_error(&self, features: &[Vec<f64>]) -> Vec<f64>;
}

pub fn estimated_mae(predicted_absolute_errors: &[f64]) -> f64 {
    if predicted_absolute_errors.is_empty() {
        return 0.0;
    }
    predicted_absolute_errors.iter().sum::<f64>() / predicted_absolute_errors.len() as f64
}

pub fn estimated_mse(predicted_squared_errors: &[f64]) -> f64 {
    if predicted_squared_errors.is_empty() {
        return 0.0;
    }
    predicted_squared_errors.iter().sum::<f64>() / predicted_squared_errors.len() as f64
}

pub fn estimated_rmse(predicted_squared_errors: &[f64]) -> f64 {
    estimated_mse(predicted_squared_errors).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_score_perfectly() {
        let y_true = vec![true, true, false, false];
        let y_pred = vec![true, true, false, false];
        let cm = ConfusionMatrix::from_labels(&y_true, &y_pred);
        assert!((cm.accuracy() - 1.0).abs() < 1e-9);
        assert!((cm.f1() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn roc_auc_separable_case_is_one() {
        let y_true = vec![false, false, true, true];
        let y_score = vec![0.1, 0.2, 0.8, 0.9];
        assert!((roc_auc(&y_true, &y_score) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cbpe_matches_realized_when_probabilities_are_certain() {
        // proba near 1.0/0.0 => CBPE confusion matrix should match the
        // realized one when predictions are (hypothetically) all correct.
        let y_pred = vec![true, true, false, false];
        let y_pred_proba = vec![0.99, 0.99, 0.01, 0.01];
        let cm = cbpe_estimate(&y_pred, &y_pred_proba);
        assert!((cm.accuracy() - 1.0).abs() < 0.02);
    }
}
