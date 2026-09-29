//! kNN out-of-distribution gate.
//!
//! Softmax confidence alone is overconfident on novel inputs, so routing
//! additionally requires the request embedding to sit near the training
//! distribution: the gate score is `1 - mean` cosine similarity to the `k`
//! nearest reference rows, and the threshold is the high quantile of
//! leave-one-out scores of the reference itself.

use safetensors::tensor::{SafeTensors, TensorView};

use super::encoder::l2_normalize;
use super::ProxyCacheError;

/// A fitted gate: reference rows (L2-normalized), neighbor count, and the
/// calibrated threshold. `threshold = f64::INFINITY` disables the gate (too
/// few reference rows to be meaningful).
#[derive(Debug, Clone)]
pub struct KnnOod {
    dim: usize,
    k: usize,
    threshold: f64,
    reference: Vec<f32>,
}

impl KnnOod {
    /// Fit the gate on training embeddings (optionally stratified by class
    /// label), subsampled to at most `max_ref` rows.
    ///
    /// * `rows` — `(embedding, class_index)` pairs; the class index is only
    ///   used for stratification.
    /// * `k` — neighbors scored per query.
    /// * `quantile` — threshold quantile over leave-one-out scores.
    pub fn fit(
        rows: &[(&[f32], usize)],
        dim: usize,
        k: usize,
        quantile: f64,
        max_ref: usize,
    ) -> Self {
        let k = k.max(1);
        if rows.is_empty() {
            return Self {
                dim,
                k,
                threshold: f64::INFINITY,
                reference: Vec::new(),
            };
        }

        // Stratify by class: equal share per class, remainder spread in row
        // order, so a dominant class cannot own the whole reference set.
        let mut by_class: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (index, (_, class)) in rows.iter().enumerate() {
            by_class.entry(*class).or_default().push(index);
        }
        let classes = by_class.len().max(1);
        let share = (max_ref / classes).max(1);
        let mut chosen: Vec<usize> = Vec::new();
        for indices in by_class.values() {
            let take = indices.len().min(share);
            // Evenly spaced stride instead of RNG: stable and cheap.
            let stride = indices.len() as f64 / take as f64;
            for step in 0..take {
                chosen.push(indices[(step as f64 * stride) as usize]);
            }
        }
        chosen.sort_unstable();
        chosen.truncate(max_ref.max(1));

        let mut reference = Vec::with_capacity(chosen.len() * dim);
        for &index in &chosen {
            let mut copy = rows[index].0.to_vec();
            l2_normalize(&mut copy);
            reference.extend_from_slice(&copy);
        }
        let reference_count = chosen.len();

        // Leave-one-out scores on the reference itself set the threshold:
        // each row is scored against the others (its own similarity is
        // masked out), sampled down when the reference is large.
        let threshold = if reference_count <= k + 1 {
            f64::INFINITY
        } else {
            self_scores(&reference, reference_count, dim, k, quantile)
        };

        Self {
            dim,
            k,
            threshold,
            reference,
        }
    }

    /// The calibrated threshold (may be infinite).
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Neighbor count.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Reference row count.
    pub fn reference_rows(&self) -> usize {
        self.reference.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Gate score for one embedding: `1 - mean(top-k cosine similarity)`.
    /// Higher means further from the training distribution.
    pub fn score(&self, embedding: &[f32]) -> f64 {
        if self.reference.is_empty() || self.threshold.is_infinite() {
            return 0.0;
        }
        let mut query = embedding.to_vec();
        l2_normalize(&mut query);
        let rows = self.reference.len() / self.dim;
        // Chunked scan over the reference set keeps the working set small.
        let chunk = 2048;
        let mut sims: Vec<f32> = Vec::with_capacity(rows);
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            for row in start..end {
                let dot: f32 = query
                    .iter()
                    .zip(&self.reference[row * self.dim..(row + 1) * self.dim])
                    .map(|(a, b)| a * b)
                    .sum();
                sims.push(dot);
            }
        }
        // Top-k mean via partial selection.
        let k = self.k.max(1).min(sims.len());
        sims.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mean: f64 = sims[..k].iter().map(|v| *v as f64).sum::<f64>() / k as f64;
        1.0 - mean
    }

    /// Serialize the reference and threshold into a safetensors buffer.
    pub fn to_safetensors(&self) -> Result<Vec<u8>, ProxyCacheError> {
        let rows = self.reference_rows();
        let reference_bytes: Vec<u8> = self
            .reference
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let threshold_bits = self.threshold.to_bits().to_le_bytes();
        let k_bits = (self.k as u64).to_le_bytes();
        let views: Vec<(&str, TensorView)> = vec![
            (
                "X",
                TensorView::new(
                    safetensors::Dtype::F32,
                    vec![rows, self.dim],
                    &reference_bytes,
                )
                .map_err(|error| ProxyCacheError::Version(format!("ood X view: {error}")))?,
            ),
            (
                "threshold_bits",
                TensorView::new(safetensors::Dtype::U64, vec![1], &threshold_bits).map_err(
                    |error| ProxyCacheError::Version(format!("ood threshold view: {error}")),
                )?,
            ),
            (
                "k",
                TensorView::new(safetensors::Dtype::U64, vec![1], &k_bits)
                    .map_err(|error| ProxyCacheError::Version(format!("ood k view: {error}")))?,
            ),
        ];
        safetensors::serialize(views, &None)
            .map_err(|error| ProxyCacheError::Version(format!("serialize ood: {error}")))
    }

    /// Restore a gate from a safetensors buffer.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, ProxyCacheError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|error| ProxyCacheError::Version(format!("deserialize ood: {error}")))?;
        let x = tensors.tensor("X").map_err(|error| {
            ProxyCacheError::Version(format!("ood artifact missing X: {error}"))
        })?;
        let threshold = tensors.tensor("threshold_bits").map_err(|error| {
            ProxyCacheError::Version(format!("ood artifact missing threshold: {error}"))
        })?;
        let k = tensors.tensor("k").map_err(|error| {
            ProxyCacheError::Version(format!("ood artifact missing k: {error}"))
        })?;
        if x.dtype() != safetensors::Dtype::F32 || x.shape().len() != 2 {
            return Err(ProxyCacheError::Version(
                "ood X must be a rank-2 F32 tensor".into(),
            ));
        }
        let dim = x.shape()[1];
        let reference: Vec<f32> = x
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let read_u64 = |tensor: &safetensors::tensor::TensorView<'_>,
                        name: &str|
         -> Result<u64, ProxyCacheError> {
            let data = tensor.data();
            if data.len() < 8 {
                return Err(ProxyCacheError::Version(format!(
                    "ood artifact {name} is truncated"
                )));
            }
            Ok(u64::from_le_bytes(data[..8].try_into().expect("8 bytes")))
        };
        let threshold = f64::from_bits(read_u64(&threshold, "threshold")?);
        let k = read_u64(&k, "k")? as usize;
        Ok(Self {
            dim,
            k,
            threshold,
            reference,
        })
    }
}

/// Leave-one-out scores of every reference row against the rest, returning
/// the configured quantile. With more than 2000 rows a random-ish stride
/// sample keeps the O(r²·d) scan bounded.
fn self_scores(reference: &[f32], rows: usize, dim: usize, k: usize, quantile: f64) -> f64 {
    let stride = rows.div_ceil(2000);
    let mut scores = Vec::new();
    for row in (0..rows).step_by(stride) {
        scores.push(loo_score(reference, rows, dim, k, row));
    }
    if scores.is_empty() {
        return f64::INFINITY;
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = (((scores.len() as f64) * quantile).floor() as usize)
        .saturating_sub(1)
        .min(scores.len() - 1);
    scores[index]
}

/// Score of reference row `row` against all other reference rows.
fn loo_score(reference: &[f32], rows: usize, dim: usize, k: usize, row: usize) -> f64 {
    let query = &reference[row * dim..(row + 1) * dim];
    let mut sims: Vec<f32> = Vec::with_capacity(rows.saturating_sub(1));
    for other in 0..rows {
        if other == row {
            continue;
        }
        let dot: f32 = query
            .iter()
            .zip(&reference[other * dim..(other + 1) * dim])
            .map(|(a, b)| a * b)
            .sum();
        sims.push(dot);
    }
    if sims.is_empty() {
        return f64::INFINITY;
    }
    let k = k.min(sims.len());
    sims.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    1.0 - sims[..k].iter().map(|v| *v as f64).sum::<f64>() / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(v: Vec<f32>) -> Vec<f32> {
        let mut v = v;
        l2_normalize(&mut v);
        v
    }

    fn rows_of(storage: &[Vec<f32>], classes: impl Fn(usize) -> usize) -> Vec<(&[f32], usize)> {
        storage
            .iter()
            .enumerate()
            .map(|(index, vector)| (vector.as_slice(), classes(index)))
            .collect()
    }

    #[test]
    fn novel_inputs_score_higher_than_in_distribution() {
        // Two dense clusters along the axes; the query sits far from both.
        let mut vectors = Vec::new();
        for i in 0..40 {
            let jitter = (i as f32 % 5.0) * 0.01;
            vectors.push(vec![1.0 + jitter, jitter]);
        }
        for i in 0..40 {
            let jitter = (i as f32 % 5.0) * 0.01;
            vectors.push(vec![jitter, 1.0 + jitter]);
        }
        let storage: Vec<Vec<f32>> = vectors.into_iter().map(normalized).collect();
        let rows = rows_of(&storage, |index| if index < 40 { 0 } else { 1 });
        let gate = KnnOod::fit(&rows, 2, 10, 0.99, 5000);
        assert!(gate.threshold().is_finite());

        let in_distribution = gate.score(&normalized(vec![1.0, 0.02]));
        let novel = gate.score(&normalized(vec![0.71, -0.71]));
        assert!(
            novel > gate.threshold(),
            "novel {novel} should exceed threshold {}",
            gate.threshold()
        );
        assert!(
            in_distribution <= gate.threshold(),
            "in-distribution {in_distribution} should pass"
        );
        assert!(novel > in_distribution);
    }

    #[test]
    fn tiny_reference_disables_the_gate() {
        let storage = vec![normalized(vec![1.0, 0.0]), normalized(vec![0.0, 1.0])];
        let rows = rows_of(&storage, |index| index);
        let gate = KnnOod::fit(&rows, 2, 10, 0.99, 5000);
        assert!(gate.threshold().is_infinite());
        assert_eq!(gate.score(&normalized(vec![1.0, 0.0])), 0.0);
    }

    #[test]
    fn stratification_caps_dominant_class() {
        let mut vectors = Vec::new();
        for i in 0..100 {
            vectors.push(vec![1.0 + (i as f32) * 0.001, 0.0]);
        }
        for i in 0..10 {
            vectors.push(vec![0.0, 1.0 + (i as f32) * 0.001]);
        }
        let storage: Vec<Vec<f32>> = vectors.into_iter().map(normalized).collect();
        let rows = rows_of(&storage, |index| if index < 100 { 0 } else { 1 });
        let gate = KnnOod::fit(&rows, 2, 5, 0.99, 50);
        // Equal shares: 25 per class even though class 0 has 10× the rows.
        assert_eq!(gate.reference_rows(), 35);
    }

    #[test]
    fn safetensors_roundtrip() {
        let mut storage: Vec<Vec<f32>> = Vec::new();
        for i in 0..30 {
            storage.push(normalized(vec![1.0 + (i as f32) * 0.01, 0.1]));
        }
        let rows = rows_of(&storage, |_| 0);
        let gate = KnnOod::fit(&rows, 2, 5, 0.99, 100);
        let bytes = gate.to_safetensors().unwrap();
        let restored = KnnOod::from_safetensors(&bytes).unwrap();
        assert_eq!(restored.reference_rows(), gate.reference_rows());
        assert!((restored.threshold() - gate.threshold()).abs() < 1e-9);
        assert_eq!(restored.k(), gate.k());
        let query = normalized(vec![1.0, 0.12]);
        assert!((restored.score(&query) - gate.score(&query)).abs() < 1e-6);
    }
}
