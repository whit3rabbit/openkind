//! MLX loader and engine entry point for the JEV-27B-VL 8-bit profile
//! (feature `mlx`, macOS arm64).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::families::support::{verify_digest, BoundedFamilyEngine, FamilyError, FamilyLimits};
use crate::qwen35::mlx::runtime::{MlxRuntime, MlxRuntimeConfig, SharedMlxRuntime};

use super::qwen35_quantized::{read_safetensors, NormConvention, QuantizedQwen35};
use super::{pins, JevConfig, JevEngine, JevForward, DECODER_PREFIX, LM_HEAD, VISION_PREFIX};

/// Runs the quantized backbone under the process-wide MLX execution lock.
struct MlxJevForward {
    backbone: QuantizedQwen35,
    runtime: SharedMlxRuntime,
}

impl JevForward for MlxJevForward {
    fn option_logits(&self, input_ids: &[u32], token_ids: &[u32]) -> Result<Vec<f64>, FamilyError> {
        self.runtime.execute(|| {
            let hidden = self.backbone.forward_last_hidden(input_ids)?;
            self.backbone.logits_for_tokens(&hidden, token_ids)
        })?
    }
}

/// Check one pinned file's size, then its SHA-256, in place.
fn verify_pinned(root: &Path, (name, size, digest): (&str, u64, &str)) -> Result<(), FamilyError> {
    let path = root.join(name);
    let length = std::fs::metadata(&path)
        .map_err(|source| FamilyError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    if length != size {
        return Err(FamilyError::ContractMismatch {
            field: "checkpoint_size",
            expected: format!("{name} is {size} bytes"),
            actual: format!("{length} bytes"),
        });
    }
    verify_digest(&path, digest)
}

/// Load the pinned JEV-27B-VL-MLX-8bit checkpoint under `model_root` and
/// return a bounded decision engine.
///
/// Every pinned file is size- and digest-verified in place before any tensor
/// is read, and the loaded tensors must match the layout contract exactly.
pub fn load(
    model_root: impl AsRef<Path>,
    limits: FamilyLimits,
) -> Result<BoundedFamilyEngine, FamilyError> {
    let root = model_root.as_ref();
    for file in [pins::CONFIG, pins::INDEX, pins::TOKENIZER]
        .into_iter()
        .chain(pins::SHARDS)
    {
        verify_pinned(root, file)?;
    }
    let config_path = root.join(pins::CONFIG.0);
    let config =
        JevConfig::parse(
            &std::fs::read(&config_path).map_err(|source| FamilyError::Io {
                path: config_path.clone(),
                source,
            })?,
        )?;
    let tokenizer = Tokenizer::from_file(root.join(pins::TOKENIZER.0))
        .map_err(|error| FamilyError::Tokenizer(error.to_string()))?;

    let runtime: SharedMlxRuntime = Arc::new(MlxRuntime::new(MlxRuntimeConfig::default())?);
    let backbone = runtime.execute(|| -> Result<QuantizedQwen35, FamilyError> {
        let mut tensors = BTreeMap::new();
        for (name, _, _) in pins::SHARDS {
            read_safetensors(&root.join(name), &[VISION_PREFIX], &mut tensors)?;
        }
        let backbone = QuantizedQwen35::new(
            tensors,
            config.geometry,
            config.quant,
            // The converted checkpoint stores the `+1` already added.
            NormConvention::Folded,
            DECODER_PREFIX,
            LM_HEAD,
        );
        backbone.validate_layout(config.vocab_size)?;
        Ok(backbone)
    })??;
    let forward: Arc<dyn JevForward> = Arc::new(MlxJevForward { backbone, runtime });
    JevEngine::bounded(forward, tokenizer, config.decision, limits)
}
