//! Compiled-in dataset definitions: the curated allowlist `dataset pin` can
//! resolve and the identity the checked-in registry must match. Definitions
//! carry no digests; `dataset pin` resolves upstream revisions, downloads the
//! referenced shards, and emits a complete registry entry for review.

use std::collections::BTreeMap;

use crate::registry::{Gated, TemplateRef};

/// Provenance of the ported request templates: the MIT-licensed harness
/// published with arXiv 2609.37647, pinned to the reviewed commit.
pub fn template() -> TemplateRef {
    TemplateRef {
        id: "jev-benchmarking".to_owned(),
        version: 1,
        source: "https://github.com/AppliedMachineLearning-Lab/jev-benchmarking@6bbdeb33474849b6de2f0cccc9f5e19756abd67e".to_owned(),
    }
}

pub struct DatasetDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub hf_repo: &'static str,
    /// (materialization split, upstream split); `eval` first.
    pub splits: [(&'static str, &'static str); 2],
    pub configs: &'static [&'static str],
    pub license: &'static str,
    pub gated: Gated,
    pub access_note: &'static str,
    pub task_family: &'static str,
    pub primitives: &'static [&'static str],
}

/// The curated datasets this build can pin and pull. The core evaluation set
/// is fully open: no gated or manual-approval dataset is included, and no
/// entry here authorizes redistribution of upstream bytes.
pub const DATASET_DEFINITIONS: &[DatasetDefinition] = &[
    DatasetDefinition {
        name: "sst2",
        description: "Binary sentence sentiment (GLUE SST-2).",
        hf_repo: "stanfordnlp/sst2",
        splits: [("eval", "validation"), ("dev", "train")],
        configs: &["default"],
        license: "unstated on the dataset card; internal evaluation only",
        gated: Gated::None,
        access_note: "",
        task_family: "text-classification",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "ag_news",
        description: "News topic classification, 4 classes.",
        hf_repo: "fancyzhx/ag_news",
        splits: [("eval", "test"), ("dev", "train")],
        configs: &["default"],
        license: "unstated on the dataset card; internal evaluation only",
        gated: Gated::None,
        access_note: "",
        task_family: "text-classification",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "banking77",
        description: "Banking customer-query intent, 77 classes.",
        hf_repo: "mteb/banking77",
        splits: [("eval", "test"), ("dev", "train")],
        configs: &["default"],
        license: "MIT",
        gated: Gated::None,
        access_note: "",
        task_family: "intent-routing",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "clinc150",
        description: "Virtual-assistant intent, 150 intents + out-of-scope.",
        hf_repo: "clinc/clinc_oos",
        splits: [("eval", "test"), ("dev", "validation")],
        configs: &["plus"],
        license: "CC-BY-3.0",
        gated: Gated::None,
        access_note: "",
        task_family: "intent-routing",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "arc",
        description: "Grade-school science QA (Challenge + Easy), 3-5 options.",
        hf_repo: "allenai/ai2_arc",
        splits: [("eval", "test"), ("dev", "validation")],
        configs: &["ARC-Challenge", "ARC-Easy"],
        license: "CC-BY-SA-4.0",
        gated: Gated::None,
        access_note: "",
        task_family: "multiple-choice",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "hellaswag",
        description: "Commonsense sentence completion, 4 options.",
        hf_repo: "Rowan/hellaswag",
        splits: [("eval", "validation"), ("dev", "train")],
        configs: &["default"],
        license: "MIT (dataset card); underlying corpus terms not detailed",
        gated: Gated::None,
        access_note: "",
        task_family: "multiple-choice",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "winogrande",
        description: "Commonsense pronoun/blank resolution, 2 options.",
        hf_repo: "allenai/winogrande",
        splits: [("eval", "validation"), ("dev", "train")],
        configs: &["winogrande_xl"],
        license: "unstated on the dataset card; internal evaluation only",
        gated: Gated::None,
        access_note: "",
        task_family: "multiple-choice",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "commonsense_qa",
        description: "Commonsense QA, 5 options.",
        hf_repo: "tau/commonsense_qa",
        splits: [("eval", "validation"), ("dev", "train")],
        configs: &["default"],
        license: "MIT",
        gated: Gated::None,
        access_note: "",
        task_family: "multiple-choice",
        primitives: &["choice"],
    },
    DatasetDefinition {
        name: "paws",
        description: "Adversarial paraphrase identification (high word overlap).",
        hf_repo: "google-research-datasets/paws",
        splits: [("eval", "test"), ("dev", "validation")],
        configs: &["labeled_final"],
        license: "custom terms on the dataset card; review before broader use",
        gated: Gated::None,
        access_note: "",
        task_family: "nli",
        primitives: &["noul"],
    },
    DatasetDefinition {
        name: "boolq",
        description: "Yes/no questions over Wikipedia passages.",
        hf_repo: "google/boolq",
        splits: [("eval", "validation"), ("dev", "train")],
        configs: &["default"],
        license: "CC-BY-SA-3.0",
        gated: Gated::None,
        access_note: "",
        task_family: "reading-comprehension",
        primitives: &["noul"],
    },
    DatasetDefinition {
        name: "stsb",
        description: "Semantic textual similarity on a 0-5 scale.",
        hf_repo: "sentence-transformers/stsb",
        splits: [("eval", "test"), ("dev", "validation")],
        configs: &["default"],
        license: "unstated on the dataset card; internal evaluation only",
        gated: Gated::None,
        access_note: "",
        task_family: "rubric-scoring",
        primitives: &["score"],
    },
    DatasetDefinition {
        name: "sst5",
        description: "Fine-grained sentence sentiment, 5 ordered levels.",
        hf_repo: "SetFit/sst5",
        splits: [("eval", "test"), ("dev", "validation")],
        configs: &["default"],
        license: "unstated on the dataset card; internal evaluation only",
        gated: Gated::None,
        access_note: "",
        task_family: "rubric-scoring",
        primitives: &["score"],
    },
];

pub fn find_definition(name: &str) -> Option<&'static DatasetDefinition> {
    DATASET_DEFINITIONS.iter().find(|def| def.name == name)
}

/// Split map for a definition, for registry-entry construction.
pub fn definition_splits(def: &DatasetDefinition) -> BTreeMap<String, String> {
    def.splits
        .iter()
        .map(|(ours, upstream)| ((*ours).to_owned(), (*upstream).to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::valid_name;

    #[test]
    fn definitions_are_well_formed() {
        let mut names = Vec::new();
        for def in DATASET_DEFINITIONS {
            assert!(valid_name(def.name), "bad name {}", def.name);
            assert_eq!(def.splits[0].0, "eval");
            assert_eq!(def.splits[1].0, "dev");
            assert!(!def.configs.is_empty());
            assert!(!def.primitives.is_empty());
            names.push(def.name);
        }
        names.sort_unstable();
        for pair in names.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
        assert_eq!(names.len(), 12);
    }
}
