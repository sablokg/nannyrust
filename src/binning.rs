//! Histogram binning shared by the continuous Jensen-Shannon and Hellinger
//! implementations. NannyML bins continuous features using Doane's formula
//! (an outlier-robust refinement of Sturges' rule), unless the feature has
//! few unique values, in which case each unique value becomes its own bin.

/*
Gaurav Sablok
gsablok@proton.me
 */

/// Sample skewness (Fisher-Pearson, uncorrected), used by Doane's formula.
fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let std = m2.sqrt();
    if std == 0.0 {
        0.0
    } else {
        m3 / std.powi(3)
    }
}

/// Number of histogram bins via Doane's formula:
/// k = 1 + log2(n) + log2(1 + |g1| / sigma_g1)
pub fn doane_bin_count(data: &[f64]) -> usize {
    let n = data.len() as f64;
    if n < 2.0 {
        return 1;
    }
    let g1 = skewness(data);
    let sigma_g1 = (6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))).sqrt();
    let k = 1.0 + n.log2() + (1.0 + g1.abs() / sigma_g1).log2();
    k.round().max(1.0) as usize
}

/// A set of bin edges (len = n_bins + 1) built from the reference sample.
/// Mirrors NannyML: if the reference has few unique values (< 10% of the
/// sample size, capped at 50), each unique value becomes its own bin;
/// otherwise use Doane's rule over the reference's min/max range.
pub fn reference_bin_edges(reference: &[f64]) -> Vec<f64> {
    let mut uniques: Vec<f64> = reference.to_vec();
    uniques.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniques.dedup();

    let n = reference.len();
    let unique_threshold = ((n as f64) * 0.10).min(50.0);

    if (uniques.len() as f64) <= unique_threshold {
        // Each unique value is its own bin: build edges at midpoints,
        // with outer edges extended slightly beyond the extremes.
        if uniques.len() == 1 {
            let v = uniques[0];
            return vec![v - 0.5, v + 0.5];
        }
        let mut edges = Vec::with_capacity(uniques.len() + 1);
        edges.push(uniques[0] - (uniques[1] - uniques[0]) / 2.0);
        for w in uniques.windows(2) {
            edges.push((w[0] + w[1]) / 2.0);
        }
        let last_two = &uniques[uniques.len() - 2..];
        edges.push(uniques[uniques.len() - 1] + (last_two[1] - last_two[0]) / 2.0);
        return edges;
    }

    let k = doane_bin_count(reference).max(1);
    let min = uniques[0];
    let max = uniques[uniques.len() - 1];
    if (max - min).abs() < f64::EPSILON {
        return vec![min - 0.5, max + 0.5];
    }
    let width = (max - min) / k as f64;
    (0..=k).map(|i| min + width * i as f64).collect()
}

/// Relative frequency of `sample` across the given bin `edges`.
/// Values below the first edge / above the last edge are clamped into the
/// outermost bins, matching np.histogram's default-adjacent behavior closely
/// enough for drift-monitoring purposes.
pub fn relative_frequencies(sample: &[f64], edges: &[f64]) -> Vec<f64> {
    let n_bins = edges.len() - 1;
    let mut counts = vec![0usize; n_bins];
    for &v in sample {
        let mut idx = match edges.binary_search_by(|e| e.partial_cmp(&v).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }
    let total = sample.len() as f64;
    if total == 0.0 {
        return vec![0.0; n_bins];
    }
    counts.into_iter().map(|c| c as f64 / total).collect()
}
