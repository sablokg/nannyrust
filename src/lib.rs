//! Rust port of NannyML's core metrics.
//!
//! Covers:
//! - Univariate drift detection (continuous): Kolmogorov-Smirnov, Wasserstein,
//!   Jensen-Shannon, Hellinger
//! - Univariate drift detection (categorical): Chi-squared, L-Infinity,
//!   Jensen-Shannon, Hellinger
//! - Performance metrics used by CBPE/DLE-style estimation: accuracy,
//!   precision, recall, F1, specificity, ROC AUC, and a CBPE-style
//!   probabilistic confusion-matrix estimator.
//!
//! This does not vendor NannyML's Python source; each function is a
//! from-scratch reimplementation of the documented algorithm (see
//! https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html).
//! No compiler was available in this sandbox to build/test this crate —
//! review before relying on it in production, and run `cargo test`.

/*
Gaurav Sablok
gsablok@proton.me
 */

pub mod binning;
pub mod categorical;
pub mod continuous;
pub mod performance;

pub use categorical::CategoricalDriftMethod;
pub use continuous::ContinuousDriftMethod;
