//! Multinomial logistic-regression student.
//!
//! `predict_proba` is a softmax over `X·W + b`. `fit` runs full-batch Adam on
//! weighted soft-target cross-entropy with early stopping on a held-out
//! validation slice; the fitted student learns the teacher's uncertainty
//! because the targets are full probability distributions by default.

use ndarray::Array2;
use safetensors::tensor::{SafeTensors, TensorView};

use super::ProxyCacheError;

/// Optimizer/report state of one fit.
#[derive(Debug, Clone)]
pub struct FitReport {
    /// Epochs actually run (early stopping may cut the budget).
    pub epochs: usize,
    /// Final weighted training cross-entropy.
    pub train_loss: f64,
    /// Best weighted validation cross-entropy (`None` when no split held out).
    pub val_loss: Option<f64>,
}

/// A linear student: `W` is `(dim, classes)` row-major, `b` is `(classes,)`.
#[derive(Debug, Clone)]
pub struct LinearStudent {
    dim: usize,
    classes: usize,
    w: Vec<f64>,
    b: Vec<f64>,
}

impl LinearStudent {
    /// A zero-initialized student (uniform softmax).
    pub fn new(dim: usize, classes: usize) -> Self {
        Self {
            dim,
            classes,
            w: vec![0.0; dim * classes],
            b: vec![0.0; classes],
        }
    }

    /// Number of input dimensions.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of output classes.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Softmax probabilities for one embedding.
    pub fn predict_proba(&self, x: &[f32]) -> Vec<f64> {
        let mut logits = self.b.clone();
        for (index, &value) in x.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let value = value as f64;
            for (class, logit) in logits.iter_mut().enumerate() {
                *logit += value * self.w[index * self.classes + class];
            }
        }
        softmax(&logits)
    }

    /// Softmax probabilities for a batch of embeddings (rows of `xs`).
    pub fn predict_proba_batch(&self, xs: &Array2<f64>) -> Array2<f64> {
        let n = xs.nrows();
        let xw = xs.dot(
            &Array2::from_shape_vec((self.dim, self.classes), self.w.clone())
                .expect("student weight shape"),
        );
        let mut out = Array2::<f64>::zeros((n, self.classes));
        for row in 0..n {
            let mut logits = vec![0.0; self.classes];
            for class in 0..self.classes {
                logits[class] = xw[[row, class]] + self.b[class];
            }
            let probs = softmax(&logits);
            for class in 0..self.classes {
                out[[row, class]] = probs[class];
            }
        }
        out
    }

    /// Full-batch Adam on weighted soft-target cross-entropy with early
    /// stopping.
    ///
    /// * `xs` — `(n, dim)` training embeddings.
    /// * `targets` — `(n, classes)` soft targets (rows need not sum to one;
    ///   they are used as-is so the student inherits teacher temperature).
    /// * `weights` — per-row importance weights, normalized internally to
    ///   mean one.
    /// * `lr`, `l2`, `epochs`, `patience`, `val_fraction`, `eval_every` —
    ///   fit hyperparameters.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        xs: &Array2<f64>,
        targets: &Array2<f64>,
        weights: &[f64],
        lr: f64,
        l2: f64,
        epochs: usize,
        patience: usize,
        val_fraction: f64,
        eval_every: usize,
        seed: u64,
    ) -> Result<(Self, FitReport), ProxyCacheError> {
        let n = xs.nrows();
        let dim = xs.ncols();
        let classes = targets.ncols();
        if targets.nrows() != n || weights.len() != n || classes == 0 || dim == 0 || n == 0 {
            return Err(ProxyCacheError::Contract(format!(
                "student fit shape mismatch: xs {}x{}, targets {}x{}, weights {}",
                n,
                dim,
                targets.nrows(),
                classes,
                weights.len()
            )));
        }

        // Seeded permutation; hold out a validation slice only when there is
        // enough data for one to be meaningful.
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = fastrand::Rng::with_seed(seed);
        for index in (1..order.len()).rev() {
            let swap = rng.usize(..=index);
            order.swap(index, swap);
        }
        let use_val = val_fraction > 0.0 && n >= 50;
        let n_val = if use_val {
            (((n as f64) * val_fraction).round() as usize).clamp(1, n / 2)
        } else {
            0
        };
        let val_indices: Vec<usize> = order[..n_val].to_vec();
        let train_indices: Vec<usize> = order[n_val..].to_vec();
        let n_train = train_indices.len();

        // Normalize weights to mean one over the train slice.
        let weight_sum: f64 = train_indices.iter().map(|&i| weights[i]).sum();
        if !(weight_sum.is_finite()) || weight_sum <= 0.0 {
            return Err(ProxyCacheError::Contract(
                "student fit received non-positive total weight".into(),
            ));
        }
        let mut w_norm = weights.to_vec();
        for index in train_indices.iter() {
            w_norm[*index] = weights[*index] / weight_sum * (n_train as f64);
        }

        let mut student = LinearStudent::new(dim, classes);
        let mut w = Array2::<f64>::zeros((dim, classes));
        let mut b = vec![0.0_f64; classes];
        let mut m_w = Array2::<f64>::zeros((dim, classes));
        let mut v_w = Array2::<f64>::zeros((dim, classes));
        let mut m_b = vec![0.0_f64; classes];
        let mut v_b = vec![0.0_f64; classes];

        // Gather the train slice once; every epoch works on dense gemms.
        let mut x_train = Array2::<f64>::zeros((n_train, dim));
        let mut y_train = Array2::<f64>::zeros((n_train, classes));
        let mut w_train = vec![0.0_f64; n_train];
        for (row, &i) in train_indices.iter().enumerate() {
            x_train.row_mut(row).assign(&xs.row(i));
            y_train.row_mut(row).assign(&targets.row(i));
            w_train[row] = w_norm[i];
        }

        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps = 1e-8;

        let mut best_val = f64::INFINITY;
        let mut best_w = w.clone();
        let mut best_b = b.clone();
        let mut best_epoch = 0usize;
        let mut bad_evaluations = 0usize;
        let mut epochs_run = 0usize;
        let mut train_loss = 0.0;

        for epoch in 1..=epochs.max(1) {
            epochs_run = epoch;

            // Forward: P = softmax(X·W + b) over the train slice.
            let logits = x_train.dot(&w);
            let mut p = Array2::<f64>::zeros((n_train, classes));
            for row in 0..n_train {
                let mut row_logits = vec![0.0; classes];
                for class in 0..classes {
                    row_logits[class] = logits[[row, class]] + b[class];
                }
                let probs = softmax(&row_logits);
                for class in 0..classes {
                    p[[row, class]] = probs[class];
                }
            }

            // Gradient: G = wt·(P − Y)/n_train; gW = Xᵀ·G + l2·W; gb = colsum(G).
            let mut g = Array2::<f64>::zeros((n_train, classes));
            let mut loss = 0.0;
            for row in 0..n_train {
                let weight = w_train[row] / (n_train as f64);
                for class in 0..classes {
                    g[[row, class]] = weight * (p[[row, class]] - y_train[[row, class]]);
                    loss -= w_train[row] * p[[row, class]].max(1e-12).ln() * y_train[[row, class]];
                }
            }
            train_loss = loss / (n_train as f64);
            let mut g_w = x_train.t().dot(&g);
            for class in 0..classes {
                for d in 0..dim {
                    g_w[[d, class]] += l2 * w[[d, class]];
                }
            }
            let mut g_b = vec![0.0_f64; classes];
            for row in 0..n_train {
                for class in 0..classes {
                    g_b[class] += g[[row, class]];
                }
            }

            // Adam update (bias-corrected).
            let t = epoch as f64;
            let bc1 = 1.0 - beta1.powf(t);
            let bc2 = 1.0 - beta2.powf(t);
            for class in 0..classes {
                for d in 0..dim {
                    let grad = g_w[[d, class]];
                    m_w[[d, class]] = beta1 * m_w[[d, class]] + (1.0 - beta1) * grad;
                    v_w[[d, class]] = beta2 * v_w[[d, class]] + (1.0 - beta2) * grad * grad;
                    let m_hat = m_w[[d, class]] / bc1;
                    let v_hat = v_w[[d, class]] / bc2;
                    w[[d, class]] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
                let grad = g_b[class];
                m_b[class] = beta1 * m_b[class] + (1.0 - beta1) * grad;
                v_b[class] = beta2 * v_b[class] + (1.0 - beta2) * grad * grad;
                let m_hat = m_b[class] / bc1;
                let v_hat = v_b[class] / bc2;
                b[class] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            // Early stopping on the weighted validation loss.
            if use_val && epoch % eval_every.max(1) == 0 {
                let val = weighted_val_loss(&w, &b, xs, targets, weights, &val_indices, classes);
                if val + 1e-5 < best_val {
                    best_val = val;
                    best_w = w.clone();
                    best_b = b.clone();
                    best_epoch = epoch;
                    bad_evaluations = 0;
                } else {
                    bad_evaluations += 1;
                    if bad_evaluations >= patience.max(1) {
                        break;
                    }
                }
            } else if !use_val {
                // No validation slice: keep the last parameters.
                best_w = w.clone();
                best_b = b.clone();
                best_epoch = epoch;
            }
        }

        student.w = best_w.into_raw_vec_and_offset().0;
        student.b = best_b;
        Ok((
            student,
            FitReport {
                epochs: epochs_run.max(best_epoch),
                train_loss,
                val_loss: if use_val { Some(best_val) } else { None },
            },
        ))
    }

    /// Serialize `W` and `b` into a safetensors byte buffer.
    pub fn to_safetensors(&self) -> Result<Vec<u8>, ProxyCacheError> {
        let w_bytes: Vec<u8> = self.w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = self.b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let views: Vec<(&str, TensorView)> = vec![
            (
                "W",
                TensorView::new(
                    safetensors::Dtype::F64,
                    vec![self.dim, self.classes],
                    &w_bytes,
                )
                .map_err(|error| ProxyCacheError::Version(format!("student W view: {error}")))?,
            ),
            (
                "b",
                TensorView::new(safetensors::Dtype::F64, vec![self.classes], &b_bytes).map_err(
                    |error| ProxyCacheError::Version(format!("student b view: {error}")),
                )?,
            ),
        ];
        safetensors::serialize(views, &None)
            .map_err(|error| ProxyCacheError::Version(format!("serialize student: {error}")))
    }

    /// Restore a student from a safetensors buffer with `W` and `b` keys.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, ProxyCacheError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|error| ProxyCacheError::Version(format!("deserialize student: {error}")))?;
        let w_tensor = tensors.tensor("W").map_err(|error| {
            ProxyCacheError::Version(format!("student artifact missing W: {error}"))
        })?;
        let b_tensor = tensors.tensor("b").map_err(|error| {
            ProxyCacheError::Version(format!("student artifact missing b: {error}"))
        })?;
        if w_tensor.dtype() != safetensors::Dtype::F64 || w_tensor.shape().len() != 2 {
            return Err(ProxyCacheError::Version(
                "student W must be a rank-2 F64 tensor".into(),
            ));
        }
        let dim = w_tensor.shape()[0];
        let classes = w_tensor.shape()[1];
        let w: Vec<f64> = w_tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk))
            .collect();
        let b: Vec<f64> = b_tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk))
            .collect();
        if w.len() != dim * classes || b.len() != classes {
            return Err(ProxyCacheError::Version(
                "student artifact shape does not match its data".into(),
            ));
        }
        Ok(Self { dim, classes, w, b })
    }
}

/// Numerically stable softmax.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if !(sum.is_finite()) || sum <= 0.0 {
        // Degenerate logits: fall back to uniform rather than NaN.
        return vec![1.0 / logits.len().max(1) as f64; logits.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

fn weighted_val_loss(
    w: &Array2<f64>,
    b: &[f64],
    xs: &Array2<f64>,
    targets: &Array2<f64>,
    weights: &[f64],
    indices: &[usize],
    classes: usize,
) -> f64 {
    if indices.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    for &i in indices {
        let x_row = xs.row(i);
        let mut logits = vec![0.0; classes];
        for (d, &value) in x_row.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            for class in 0..classes {
                logits[class] += value * w[[d, class]];
            }
        }
        for class in 0..classes {
            logits[class] += b[class];
        }
        let probs = softmax(&logits);
        for class in 0..classes {
            total -= weights[i] * probs[class].max(1e-12).ln() * targets[[i, class]];
        }
        weight_sum += weights[i];
    }
    if weight_sum <= 0.0 {
        return f64::INFINITY;
    }
    total / weight_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn flat(rows: &[Vec<f64>]) -> Array2<f64> {
        let n = rows.len();
        let d = rows[0].len();
        Array2::from_shape_vec((n, d), rows.iter().flatten().cloned().collect()).unwrap()
    }

    #[test]
    fn zero_student_predicts_uniform() {
        let student = LinearStudent::new(4, 3);
        let probs = student.predict_proba(&[0.0; 4]);
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn fit_learns_separable_soft_targets() {
        // Two well-separated clusters; teacher is nearly peaked per cluster.
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for i in 0..120 {
            if i % 2 == 0 {
                rows.push(vec![1.0, 0.0]);
                targets.push(vec![0.95, 0.05]);
            } else {
                rows.push(vec![0.0, 1.0]);
                targets.push(vec![0.05, 0.95]);
            }
        }
        let xs = flat(&rows);
        let ys = flat(&targets);
        let weights = vec![1.0; 120];
        let (student, report) =
            LinearStudent::fit(&xs, &ys, &weights, 0.05, 1e-6, 500, 4, 0.0, 25, 0).unwrap();
        let a = student.predict_proba(&[1.0, 0.0]);
        let b = student.predict_proba(&[0.0, 1.0]);
        assert!(a[0] > 0.9, "cluster A probs {a:?}");
        assert!(b[1] > 0.9, "cluster B probs {b:?}");
        assert!(report.train_loss < 0.2, "loss {}", report.train_loss);
    }

    #[test]
    fn fit_learns_uncertainty_not_just_argmax() {
        // One ambiguous cluster the teacher splits ~50/50.
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for i in 0..160 {
            if i % 2 == 0 {
                rows.push(vec![1.0, 0.0]);
                targets.push(vec![0.98, 0.02]);
            } else {
                rows.push(vec![0.0, 1.0]);
                targets.push(vec![0.5, 0.5]);
            }
        }
        let xs = flat(&rows);
        let ys = flat(&targets);
        let weights = vec![1.0; 160];
        let (student, _) =
            LinearStudent::fit(&xs, &ys, &weights, 0.05, 1e-6, 800, 4, 0.0, 25, 0).unwrap();
        let ambiguous = student.predict_proba(&[0.0, 1.0]);
        assert!(
            (ambiguous[0] - 0.5).abs() < 0.15,
            "ambiguous cluster should stay near 0.5: {ambiguous:?}"
        );
        let peaked = student.predict_proba(&[1.0, 0.0]);
        assert!(peaked[0] > 0.9, "peaked cluster {peaked:?}");
    }

    #[test]
    fn weights_change_the_fit() {
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..100 {
            rows.push(vec![1.0, 0.0]);
            weights.push(1.0);
            if i < 80 {
                targets.push(vec![0.9, 0.1]);
            } else {
                targets.push(vec![0.1, 0.9]);
                weights[i] = 20.0;
            }
        }
        for _ in 0..100 {
            rows.push(vec![0.0, 1.0]);
            targets.push(vec![0.1, 0.9]);
            weights.push(1.0);
        }
        let xs = flat(&rows);
        let ys = flat(&targets);
        let (student, _) =
            LinearStudent::fit(&xs, &ys, &weights, 0.05, 1e-6, 600, 4, 0.0, 25, 0).unwrap();
        // Heavy weight on the minority [1,0]→[0.1,0.9] rows pulls that
        // cluster's prediction away from the majority label.
        let a = student.predict_proba(&[1.0, 0.0]);
        assert!(
            a[1] > 0.3,
            "weighted minority rows should move the prediction: {a:?}"
        );
    }

    #[test]
    fn early_stopping_with_validation_slice() {
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for i in 0..200 {
            let cluster = i % 3;
            let mut row = vec![0.0; 3];
            row[cluster] = 1.0;
            rows.push(row.clone());
            let mut target = vec![0.02; 3];
            target[cluster] = 0.94;
            targets.push(target);
        }
        let xs = flat(&rows);
        let ys = flat(&targets);
        let weights = vec![1.0; 200];
        let (student, report) =
            LinearStudent::fit(&xs, &ys, &weights, 0.05, 1e-6, 2000, 2, 0.1, 10, 3).unwrap();
        assert!(report.val_loss.is_some());
        assert!(report.epochs < 2000, "early stopping should cut epochs");
        let probs = student.predict_proba(&[0.0, 1.0, 0.0]);
        assert!(probs[1] > 0.9);
    }

    #[test]
    fn safetensors_roundtrip() {
        let mut student = LinearStudent::new(5, 3);
        for (index, value) in student.w.iter_mut().enumerate() {
            *value = (index as f64) * 0.25;
        }
        student.b[1] = 0.5;
        let bytes = student.to_safetensors().unwrap();
        let restored = LinearStudent::from_safetensors(&bytes).unwrap();
        assert_eq!(restored.dim(), 5);
        assert_eq!(restored.classes(), 3);
        assert_eq!(restored.w, student.w);
        assert_eq!(restored.b, student.b);
    }

    #[test]
    fn predict_batch_matches_single() {
        let mut student = LinearStudent::new(2, 2);
        student.w.copy_from_slice(&[0.3, -0.7, 1.1, 0.2]);
        student.b.copy_from_slice(&[0.05, -0.05]);
        let xs = array![[1.0, 2.0], [0.5, -0.5]];
        let batch = student.predict_proba_batch(&xs);
        let single_a = student.predict_proba(&[1.0, 2.0]);
        let single_b = student.predict_proba(&[0.5, -0.5]);
        assert!((batch[[0, 0]] - single_a[0]).abs() < 1e-12);
        assert!((batch[[1, 1]] - single_b[1]).abs() < 1e-12);
    }
}
