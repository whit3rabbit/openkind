//! Knockout combination for questions with more than 16 options.
//!
//! A faithful port of the reference `jevk5` `prompt.py` knockout schedule:
//! the options are split, in their given order, into near-equal groups of at
//! most 16, every group is read, and a final pass re-reads the groups' best
//! options. Finalists keep the final's distribution times the chance the
//! answer is a finalist; every other option keeps its group's share of the
//! final times its in-group probability. Ties resolve to the earlier option
//! because the sorts are stable. Beyond 256 options the schedule recurses.

use super::MAX_OPTIONS_PER_PASS;

/// Read one group of at most 16 option texts, returning the calibrated
/// distribution over that group's options in order.
pub(crate) trait PassReader {
    fn read(
        &self,
        options: &[(String, String)],
    ) -> Result<Vec<f64>, crate::families::support::FamilyError>;
}

/// Combine a distribution over every option from reads of at most 16
/// options each.
///
/// Up to 16 options the single read is returned untouched. Larger sets run
/// the knockout schedule and normalize the combined weights.
pub(crate) fn combine(
    reader: &dyn PassReader,
    options: &[(String, String)],
) -> Result<Vec<f64>, crate::families::support::FamilyError> {
    if options.len() <= super::MAX_OPTIONS_PER_PASS {
        return reader.read(options);
    }
    let weights = knockout(reader, options)?;
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(crate::families::support::FamilyError::Numerical(format!(
            "knockout combination produced a non-positive normalizer: {total}"
        )));
    }
    Ok(weights.iter().map(|weight| weight / total).collect())
}

/// Contiguous near-equal runs covering `0..count`.
fn groups(count: usize, groups_wanted: usize) -> Vec<std::ops::Range<usize>> {
    let base = count / groups_wanted;
    let extra = count % groups_wanted;
    let mut runs = Vec::with_capacity(groups_wanted);
    let mut start = 0;
    for group in 0..groups_wanted {
        let stop = start + base + usize::from(group < extra);
        runs.push(start..stop);
        start = stop;
    }
    runs
}

fn knockout(
    reader: &dyn PassReader,
    options: &[(String, String)],
) -> Result<Vec<f64>, crate::families::support::FamilyError> {
    let runs = groups(options.len(), options.len().div_ceil(MAX_OPTIONS_PER_PASS));
    let inner: Vec<Vec<f64>> = runs
        .iter()
        .map(|run| {
            let group: Vec<(String, String)> =
                run.clone().map(|index| options[index].clone()).collect();
            let probabilities = reader.read(&group)?;
            normalize(&probabilities)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The final pass keeps the top `keep` options of every group (at least
    // one), free places going to the next most likely options overall.
    let keep = (MAX_OPTIONS_PER_PASS / runs.len()).max(1);
    let ranked: Vec<Vec<usize>> = inner
        .iter()
        .map(|probabilities| {
            let mut order: Vec<usize> = (0..probabilities.len()).collect();
            order.sort_by(|&left, &right| {
                probabilities[right]
                    .total_cmp(&probabilities[left])
                    .then(left.cmp(&right))
            });
            order
        })
        .collect();
    let mut chosen: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    for (group, order) in ranked.iter().enumerate() {
        for &index in order.iter().take(keep) {
            chosen.insert((group, index));
        }
    }
    let mut rest: Vec<(usize, usize)> = ranked
        .iter()
        .enumerate()
        .flat_map(|(group, order)| order.iter().skip(keep).map(move |&index| (group, index)))
        .collect();
    rest.sort_by(|&(left_group, left_index), &(right_group, right_index)| {
        inner[left_group][left_index]
            .total_cmp(&inner[right_group][right_index])
            .reverse()
    });
    let free = MAX_OPTIONS_PER_PASS.saturating_sub(chosen.len());
    chosen.extend(rest.into_iter().take(free));

    let finalists: Vec<(String, String)> = runs
        .iter()
        .enumerate()
        .flat_map(|(group, run)| {
            chosen
                .iter()
                .filter(move |&&(chosen_group, _)| chosen_group == group)
                .map(move |&(_, index)| options[run.start + index].clone())
        })
        .collect();
    let final_distribution = combine(reader, &finalists)?;
    // Shares map each group's finalist IN-GROUP indices (ascending, matching
    // the reference `tops`) to their final-pass probabilities.
    let mut shares: Vec<Vec<(usize, f64)>> = Vec::with_capacity(runs.len());
    let mut at = 0;
    for group in 0..runs.len() {
        let tops: Vec<usize> = chosen
            .iter()
            .filter(|&&(chosen_group, _)| chosen_group == group)
            .map(|&(_, index)| index)
            .collect();
        let finals = &final_distribution[at..at + tops.len()];
        shares.push(tops.into_iter().zip(finals.iter().copied()).collect());
        at += finals.len();
    }

    // Reference formula: the chance the answer is a finalist, summed per
    // group as the group's final-pass share times its in-group finalist
    // probability.
    let in_final: f64 = inner
        .iter()
        .zip(&shares)
        .map(|(probabilities, share)| {
            let final_mass: f64 = share.iter().map(|&(_, value)| value).sum();
            let finalist_probability: f64 =
                share.iter().map(|&(index, _)| probabilities[index]).sum();
            final_mass * finalist_probability
        })
        .sum();
    let mut weights = Vec::with_capacity(options.len());
    for (group, probabilities) in inner.iter().enumerate() {
        let share = &shares[group];
        let mass: f64 = share
            .iter()
            .map(|&(_, final_probability)| final_probability)
            .sum();
        let finalist_indices: std::collections::BTreeSet<usize> =
            share.iter().map(|&(index, _)| index).collect();
        for (index, &probability) in probabilities.iter().enumerate() {
            if finalist_indices.contains(&index) {
                let final_probability = share
                    .iter()
                    .find(|&&(share_index, _)| share_index == index)
                    .map(|&(_, value)| value)
                    .unwrap_or_default();
                weights.push(final_probability * in_final);
            } else {
                weights.push(mass * probability);
            }
        }
    }
    Ok(weights)
}

fn normalize(probabilities: &[f64]) -> Result<Vec<f64>, crate::families::support::FamilyError> {
    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(crate::families::support::FamilyError::Numerical(format!(
            "group read produced a non-positive normalizer: {total}"
        )));
    }
    Ok(probabilities
        .iter()
        .map(|probability| probability / total)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct ScriptedReader {
        responses: RefCell<Vec<Vec<f64>>>,
    }

    impl PassReader for ScriptedReader {
        fn read(
            &self,
            options: &[(String, String)],
        ) -> Result<Vec<f64>, crate::families::support::FamilyError> {
            let response = self
                .responses
                .borrow_mut()
                .pop()
                .expect("more reads than scripted");
            assert_eq!(options.len(), response.len(), "pass width mismatch");
            Ok(response)
        }
    }

    fn options(count: usize) -> Vec<(String, String)> {
        (0..count)
            .map(|index| (format!("opt{index}"), format!("Option {index}")))
            .collect()
    }

    #[test]
    fn small_questions_take_a_single_untouched_read() {
        let reader = ScriptedReader {
            responses: RefCell::new(vec![vec![0.1, 0.3, 0.6]]),
        };
        let combined = combine(&reader, &options(3)).expect("combine");
        assert_eq!(combined, vec![0.1, 0.3, 0.6]);
    }

    #[test]
    fn knockout_covers_every_option_and_normalizes() {
        // 20 options -> 2 near-equal groups (10 + 10) + 1 final pass of 16
        // finalists (keep = 16 // 2 = 8 per group).
        let first = vec![0.5; 10];
        let mut second = vec![0.1; 10];
        second[3] = 10.0;
        let mut final_pass = vec![0.01; 16];
        // Finalists map in (group, ascending in-group index) order: positions
        // 0..7 are group 1's options 0..7 and positions 8..15 are group 2's
        // options 10..17, so option 13 sits at position 11.
        final_pass[11] = 4.0;
        let reader = ScriptedReader {
            responses: RefCell::new(vec![final_pass, second, first]),
        };
        let combined = combine(&reader, &options(20)).expect("combine");
        assert_eq!(combined.len(), 20);
        let total: f64 = combined.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "distribution sums to one: {total}"
        );
        // Reads were consumed in order: the two 10-wide groups, then the
        // 16-wide final.
        assert!(reader.responses.borrow().is_empty(), "all passes ran");
        // Option 13 is group 2's runaway finalist and keeps finalist mass.
        let argmax = combined
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("non-empty");
        assert_eq!(argmax, 13);
        // Group 1's unchosen tail (options 8, 9) keeps only its collapsed
        // group-share mass, the smallest of the set.
        let smallest = combined
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("non-empty");
        assert!(smallest == 8 || smallest == 9, "smallest was {smallest}");
    }

    #[test]
    fn groups_partition_in_order_with_near_equal_sizes() {
        let runs = groups(20, 2);
        assert_eq!(runs, vec![0..10, 10..20]);
        let runs = groups(33, 3);
        assert_eq!(runs, vec![0..11, 11..22, 22..33]);
        let runs = groups(17, 2);
        assert_eq!(runs, vec![0..9, 9..17]);
    }
}
