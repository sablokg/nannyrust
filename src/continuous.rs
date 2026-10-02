//! Univariate drift detection methods for continuous features.
//! Mirrors `nannyml.drift.univariate.methods` for continuous columns.

/*
Gaurav Sablok
gsablok@proton.me
 */

use crate::binning::{reference_bin_edges, relative_frequencies};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousDriftMethod {
    KolmogorovSmirnov,
    Wasserstein,
    JensenShannon,
    Hellinger,
}

/// Two-sample Kolmogorov-Smirnov statistic: the max absolute difference
/// between the empirical CDFs of `reference` and `analysis`.
/// Missing values should be dropped by the caller before calling this,
/// matching NannyML's documented behavior.
pub fn kolmogorov_smirnov(reference: &[f64], analysis: &[f64]) -> f64 {
    let mut r = reference.to_vec();
    let mut a = analysis.to_vec();
    r.sort_by(|x, y| x.partial_cmp(y).unwrap());
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let (n, m) = (r.len(), a.len());
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut all_values: Vec<f64> = r.iter().chain(a.iter()).cloned().collect();
    all_values.sort_by(|x, y| x.partial_cmp(y).unwrap());
    all_values.dedup();

    let mut max_diff = 0.0f64;
    for v in all_values {
        let cdf_r = r.partition_point(|&x| x <= v) as f64 / n as f64;
        let cdf_a = a.partition_point(|&x| x <= v) as f64 / m as f64;
        max_diff = max_diff.max((cdf_r - cdf_a).abs());
    }
    max_diff
}

/// 1-D Wasserstein (earth mover's) distance between two samples, computed
/// exactly via the CDF-integration identity:
/// W1(F, G) = integral_x |F(x) - G(x)| dx
pub fn wasserstein(reference: &[f64], analysis: &[f64]) -> f64 {
    let mut r = reference.to_vec();
    let mut a = analysis.to_vec();
    r.sort_by(|x, y| x.partial_cmp(y).unwrap());
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let (n, m) = (r.len(), a.len());
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut all_values: Vec<f64> = r.iter().chain(a.iter()).cloned().collect();
    all_values.sort_by(|x, y| x.partial_cmp(y).unwrap());
    all_values.dedup();

    if all_values.len() < 2 {
        return 0.0;
    }

    let mut area = 0.0f64;
    for w in all_values.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        let cdf_r = r.partition_point(|&x| x <= x0) as f64 / n as f64;
        let cdf_a = a.partition_point(|&x| x <= x0) as f64 / m as f64;
        area += (cdf_r - cdf_a).abs() * (x1 - x0);
    }
    area
}

/// Jensen-Shannon distance (sqrt of JS divergence) between the histograms
/// of `reference` and `analysis`, using bin edges derived from `reference`
/// via Doane's formula (see `binning` module).
pub fn jensen_shannon(reference: &[f64], analysis: &[f64]) -> f64 {
    let edges = reference_bin_edges(reference);
    let p = relative_frequencies(reference, &edges);
    let q = relative_frequencies(analysis, &edges);
    jensen_shannon_from_distributions(&p, &q)
}

/// Hellinger distance between the histograms of `reference` and `analysis`,
/// using the same binning scheme as `jensen_shannon`.
pub fn hellinger(reference: &[f64], analysis: &[f64]) -> f64 {
    let edges = reference_bin_edges(reference);
    let p = relative_frequencies(reference, &edges);
    let q = relative_frequencies(analysis, &edges);
    hellinger_from_distributions(&p, &q)
}

/// KL divergence in bits-free (natural log) form, with 0*log(0/q) := 0.
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(
            |(&pi, &qi)| {
                if pi == 0.0 {
                    0.0
                } else {
                    pi * (pi / qi).ln()
                }
            },
        )
        .sum()
}

pub(crate) fn jensen_shannon_from_distributions(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
    let divergence = 0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m);
    divergence.max(0.0).sqrt()
}

pub(crate) fn hellinger_from_distributions(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi.sqrt() - qi.sqrt()).powi(2))
        .sum();
    (0.5 * sum).sqrt()
}

pub fn compute(method: ContinuousDriftMethod, reference: &[f64], analysis: &[f64]) -> f64 {
    match method {
        ContinuousDriftMethod::KolmogorovSmirnov => kolmogorov_smirnov(reference, analysis),
        ContinuousDriftMethod::Wasserstein => wasserstein(reference, analysis),
        ContinuousDriftMethod::JensenShannon => jensen_shannon(reference, analysis),
        ContinuousDriftMethod::Hellinger => hellinger(reference, analysis),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_zero_drift() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!(kolmogorov_smirnov(&data, &data) < 1e-9);
        assert!(wasserstein(&data, &data) < 1e-9);
        assert!(jensen_shannon(&data, &data) < 1e-9);
        assert!(hellinger(&data, &data) < 1e-9);
    }

    #[test]
    fn shifted_samples_show_drift() {
        let reference: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let analysis: Vec<f64> = (0..100).map(|i| i as f64 + 50.0).collect();
        assert!(kolmogorov_smirnov(&reference, &analysis) > 0.4);
        assert!(wasserstein(&reference, &analysis) > 40.0);
        assert!(jensen_shannon(&reference, &analysis) > 0.1);
    }
}
