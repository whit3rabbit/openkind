use std::collections::{BTreeMap, BTreeSet};

use crate::families::support::FamilyError;

/// Largest option group GEV scores in one pass.
pub const GEV_MAX_GROUP: usize = 16;

/// GEV choice scoring for any option count up to 256.
///
/// Up to 16 options are scored in one pass. More are split into
/// `ceil(n / 16)` balanced contiguous groups, each scored on its own. The
/// best option of every group, topped up with the best remaining options to 16
/// finalists, is then scored together. Each option's probability is its
/// in-group probability scaled by its group's share of the final pass.
///
/// `pass` receives ascending option indices (at most 16) and returns one
/// probability per index, in order. The result sums to one and follows the
/// reference exactly, including its tie-breaks (higher probability, then lower
/// index).
pub fn gev_choice_tournament(
    option_count: usize,
    mut pass: impl FnMut(&[usize]) -> Result<Vec<f64>, FamilyError>,
) -> Result<Vec<f64>, FamilyError> {
    let mut scored = |indices: &[usize]| -> Result<Vec<f64>, FamilyError> {
        let probabilities = pass(indices)?;
        if probabilities.len() != indices.len() {
            return Err(FamilyError::InvalidInput(format!(
                "tournament pass returned {} probabilities for {} options",
                probabilities.len(),
                indices.len()
            )));
        }
        Ok(probabilities)
    };
    if option_count == 0 || option_count > super::MAX_CHOICE_OPTIONS {
        return Err(FamilyError::InvalidInput(format!(
            "choice question has {option_count} options, outside 1..=256"
        )));
    }
    if option_count <= GEV_MAX_GROUP {
        let all: Vec<usize> = (0..option_count).collect();
        return scored(&all);
    }

    let group_count = option_count.div_ceil(GEV_MAX_GROUP);
    let (size, extra) = (option_count / group_count, option_count % group_count);
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(group_count);
    let mut start = 0;
    for index in 0..group_count {
        let end = start + size + usize::from(index < extra);
        groups.push((start..end).collect());
        start = end;
    }

    let mut in_group = vec![0.0; option_count];
    for group in &groups {
        for (&option, probability) in group.iter().zip(scored(group)?) {
            in_group[option] = probability;
        }
    }
    let better = |a: usize, b: usize| in_group[b].total_cmp(&in_group[a]).then_with(|| a.cmp(&b));
    let chosen: Vec<usize> = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .copied()
                .min_by(|&a, &b| better(a, b))
                .expect("groups are non-empty")
        })
        .collect();
    let chosen_set: BTreeSet<usize> = chosen.iter().copied().collect();
    let mut rest: Vec<usize> = groups
        .iter()
        .flatten()
        .copied()
        .filter(|option| !chosen_set.contains(option))
        .collect();
    rest.sort_by(|&a, &b| better(a, b));
    let room = GEV_MAX_GROUP.saturating_sub(chosen.len());
    let mut finalists: Vec<usize> = chosen
        .iter()
        .copied()
        .chain(rest.into_iter().take(room))
        .collect();
    finalists.sort_unstable();

    let final_probabilities: BTreeMap<usize, f64> =
        finalists.iter().copied().zip(scored(&finalists)?).collect();
    let group_of: BTreeMap<usize, usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| group.iter().map(move |&option| (option, index)))
        .collect();
    let mut share = vec![0.0; group_count];
    let mut cap = vec![0.0; group_count];
    for (&option, &probability) in &final_probabilities {
        share[group_of[&option]] += probability;
        cap[group_of[&option]] += in_group[option];
    }
    let among: f64 = share.iter().zip(&cap).map(|(a, b)| a * b).sum();
    let mut combined: Vec<f64> = (0..option_count)
        .map(|option| match final_probabilities.get(&option) {
            Some(probability) => probability * among,
            None => share[group_of[&option]] * in_group[option],
        })
        .collect();
    let total: f64 = combined.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(FamilyError::Numerical(format!(
            "tournament normalizer is not finite: {total}"
        )));
    }
    for value in &mut combined {
        *value /= total;
    }
    Ok(combined)
}
