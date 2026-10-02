//! Univariate drift detection methods for categorical features.
//! Mirrors `nannyml.drift.univariate.methods` for categorical columns.

/*
Gaurav Sablok
gsablok@proton.me
 */

use crate::continuous::{hellinger_from_distributions, jensen_shannon_from_distributions};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalDriftMethod {
    Chi2,
    LInfinity,
    JensenShannon,
    Hellinger,
}

/// Result of a chi-squared homogeneity test between reference and analysis
/// category counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chi2Result {
    /// Pearson's chi-squared statistic.
    pub statistic: f64,
    /// Degrees of freedom: (num_categories - 1) for a 2-sample homogeneity
    /// test, since rows = 2 (reference, analysis) and df = (rows-1)*(cols-1).
    pub degrees_of_freedom: usize,
    /// Upper-tail p-value: P(X >= statistic) under a chi-squared
    /// distribution with `degrees_of_freedom` degrees of freedom.
    pub p_value: f64,
}

/// Build aligned relative-frequency vectors over the union of categories
/// seen in `reference` and `analysis`. New categories that only appear in
/// `analysis` get a reference frequency of 0, matching NannyML's approach
/// of treating unseen categories as a new bin with 0 reference mass.
fn aligned_distributions(reference: &[String], analysis: &[String]) -> (Vec<f64>, Vec<f64>) {
    let mut categories: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for v in reference {
        categories.entry(v.as_str()).or_insert((0, 0)).0 += 1;
    }
    for v in analysis {
        categories.entry(v.as_str()).or_insert((0, 0)).1 += 1;
    }

    let n_ref = reference.len().max(1) as f64;
    let n_ana = analysis.len().max(1) as f64;

    let mut p = Vec::with_capacity(categories.len());
    let mut q = Vec::with_capacity(categories.len());
    for (_, (ref_count, ana_count)) in categories {
        p.push(ref_count as f64 / n_ref);
        q.push(ana_count as f64 / n_ana);
    }
    (p, q)
}

/// Aligned raw counts (not frequencies), needed for the chi-squared statistic.
fn aligned_counts(reference: &[String], analysis: &[String]) -> (Vec<usize>, Vec<usize>) {
    let mut categories: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for v in reference {
        categories.entry(v.as_str()).or_insert((0, 0)).0 += 1;
    }
    for v in analysis {
        categories.entry(v.as_str()).or_insert((0, 0)).1 += 1;
    }
    let mut r = Vec::with_capacity(categories.len());
    let mut a = Vec::with_capacity(categories.len());
    for (_, (rc, ac)) in categories {
        r.push(rc);
        a.push(ac);
    }
    (r, a)
}

/// Regularized upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a).
/// Used to compute the chi-squared survival function:
/// P(X >= x) = Q(df/2, x/2).
///
/// Implementation follows the standard Numerical Recipes approach: a series
/// expansion for x < a + 1, and a continued fraction (Lentz's method) for
/// x >= a + 1, since the series converges slowly in that regime.
fn upper_incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // P(a, x) via series, then Q = 1 - P.
        1.0 - lower_gamma_series(a, x)
    } else {
        // Q(a, x) via continued fraction directly.
        upper_gamma_continued_fraction(a, x)
    }
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g=7, n=9 coefficients), standard double-precision variant.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFS[0];
        let t = x + G + 0.5;
        for (i, c) in COEFFS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

fn lower_gamma_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut n = a;
    for _ in 0..500 {
        n += 1.0;
        term *= x / n;
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn upper_gamma_continued_fraction(a: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Upper-tail p-value of a chi-squared distribution: P(X >= statistic)
/// for `df` degrees of freedom.
pub fn chi2_p_value(statistic: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }
    upper_incomplete_gamma_regularized(df as f64 / 2.0, statistic / 2.0).clamp(0.0, 1.0)
}

/// Pearson's chi-squared test for homogeneity between the reference and
/// analysis category counts (a 2-sample test: rows = {reference, analysis},
/// columns = categories). Expected counts are derived under the null
/// hypothesis that both samples are drawn from the same overall category
/// distribution.
///
/// Returns both the raw statistic and its upper-tail p-value, matching
/// NannyML's default behaviour of thresholding drift on the p-value rather
/// than the statistic itself.
pub fn chi2(reference: &[String], analysis: &[String]) -> Chi2Result {
    let (ref_counts, ana_counts) = aligned_counts(reference, analysis);
    let n_ref: usize = ref_counts.iter().sum();
    let n_ana: usize = ana_counts.iter().sum();
    let n_total = (n_ref + n_ana) as f64;

    let num_categories = ref_counts.len();
    let degrees_of_freedom = num_categories.saturating_sub(1);

    if n_total == 0.0 || num_categories == 0 {
        return Chi2Result {
            statistic: 0.0,
            degrees_of_freedom,
            p_value: 1.0,
        };
    }

    let mut statistic = 0.0f64;
    for (rc, ac) in ref_counts.iter().zip(ana_counts.iter()) {
        let row_total = (*rc + *ac) as f64;
        let expected_ref = row_total * n_ref as f64 / n_total;
        let expected_ana = row_total * n_ana as f64 / n_total;
        if expected_ref > 0.0 {
            statistic += (*rc as f64 - expected_ref).powi(2) / expected_ref;
        }
        if expected_ana > 0.0 {
            statistic += (*ac as f64 - expected_ana).powi(2) / expected_ana;
        }
    }

    let p_value = chi2_p_value(statistic, degrees_of_freedom);

    Chi2Result {
        statistic,
        degrees_of_freedom,
        p_value,
    }
}

/// L-Infinity distance: the maximum absolute difference in relative
/// frequency across all categories. Good for high-cardinality features.
pub fn l_infinity(reference: &[String], analysis: &[String]) -> f64 {
    let (p, q) = aligned_distributions(reference, analysis);
    p.iter()
        .zip(q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

pub fn jensen_shannon(reference: &[String], analysis: &[String]) -> f64 {
    let (p, q) = aligned_distributions(reference, analysis);
    jensen_shannon_from_distributions(&p, &q)
}

pub fn hellinger(reference: &[String], analysis: &[String]) -> f64 {
    let (p, q) = aligned_distributions(reference, analysis);
    hellinger_from_distributions(&p, &q)
}

/// Computes the configured drift method. For `Chi2`, this returns the
/// **p-value** (not the raw statistic), so a *smaller* value indicates more
/// drift — the opposite direction of the other three methods here. Callers
/// that need the statistic directly should call [`chi2`] instead.
pub fn compute(method: CategoricalDriftMethod, reference: &[String], analysis: &[String]) -> f64 {
    match method {
        CategoricalDriftMethod::Chi2 => chi2(reference, analysis).p_value,
        CategoricalDriftMethod::LInfinity => l_infinity(reference, analysis),
        CategoricalDriftMethod::JensenShannon => jensen_shannon(reference, analysis),
        CategoricalDriftMethod::Hellinger => hellinger(reference, analysis),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn identical_distributions_have_zero_drift() {
        let data = strs(&["a", "a", "b", "b", "c"]);
        let result = chi2(&data, &data);
        assert!(result.statistic < 1e-9);
        assert!((result.p_value - 1.0).abs() < 1e-6);
        assert!(l_infinity(&data, &data) < 1e-9);
        assert!(jensen_shannon(&data, &data) < 1e-9);
        assert!(hellinger(&data, &data) < 1e-9);
    }

    #[test]
    fn disjoint_distributions_show_max_drift() {
        let reference = strs(&["a", "a", "a", "a"]);
        let analysis = strs(&["b", "b", "b", "b"]);
        let result = chi2(&reference, &analysis);
        // Total separation should produce a very small p-value (highly significant).
        assert!(result.p_value < 0.01);
        assert!((l_infinity(&reference, &analysis) - 1.0).abs() < 1e-9);
        assert!(jensen_shannon(&reference, &analysis) > 0.8);
    }

    #[test]
    fn chi2_p_value_matches_known_values() {
        // Chi-squared with df=1, statistic=3.841 -> p ~= 0.05
        let p = chi2_p_value(3.841_458_82, 1);
        assert!((p - 0.05).abs() < 1e-4);

        // df=2, statistic=5.991 -> p ~= 0.05
        let p2 = chi2_p_value(5.991_464_55, 2);
        assert!((p2 - 0.05).abs() < 1e-4);
    }
}
