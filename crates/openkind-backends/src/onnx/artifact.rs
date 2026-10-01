//! Verify an operator-pinned ONNX export before handing any files to ort.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::OnnxError;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    files: BTreeMap<String, FilePin>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FilePin {
    size_bytes: u64,
    sha256: String,
}

fn integrity(path: &Path, message: impl Into<String>) -> OnnxError {
    OnnxError::Integrity {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn read_error(path: &Path, source: std::io::Error) -> OnnxError {
    OnnxError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn manifest_path(model: &Path) -> PathBuf {
    model.with_extension("onnx.manifest.json")
}

fn local_file(root: &Path, name: &str) -> Result<PathBuf, OnnxError> {
    let relative = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(integrity(
            root,
            format!("export path `{name}` must be relative without traversal"),
        ));
    }
    let path = root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| read_error(&path, error))?;
    if !canonical.starts_with(root) {
        return Err(integrity(
            &path,
            "export file resolves outside the model root",
        ));
    }
    Ok(path)
}

fn verify_file(path: &Path, pin: &FilePin) -> Result<(), OnnxError> {
    if pin.sha256.len() != 64 || !pin.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(integrity(
            path,
            "SHA-256 must contain exactly 64 hex digits",
        ));
    }
    let mut file = std::fs::File::open(path).map_err(|error| read_error(path, error))?;
    let metadata = file.metadata().map_err(|error| read_error(path, error))?;
    if !metadata.is_file() || metadata.len() != pin.size_bytes {
        return Err(integrity(
            path,
            format!("expected {} bytes, got {}", pin.size_bytes, metadata.len()),
        ));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let length = file
            .read(&mut buffer)
            .map_err(|error| read_error(path, error))?;
        if length == 0 {
            break;
        }
        hasher.update(&buffer[..length]);
    }
    let observed = format!("{:x}", hasher.finalize());
    if !observed.eq_ignore_ascii_case(&pin.sha256) {
        return Err(integrity(
            path,
            format!("SHA-256 mismatch: expected {}, got {observed}", pin.sha256),
        ));
    }
    Ok(())
}

pub(super) fn verify(model_path: &Path) -> Result<String, OnnxError> {
    let manifest_path = manifest_path(model_path);
    let bytes = std::fs::read(&manifest_path).map_err(|error| read_error(&manifest_path, error))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|error| integrity(&manifest_path, error.to_string()))?;
    if manifest.schema_version != 1 {
        return Err(integrity(
            &manifest_path,
            "unsupported export-manifest schema version",
        ));
    }
    let root = model_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .map_err(|error| read_error(model_path, error))?;
    let name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| integrity(model_path, "graph file name must be UTF-8"))?;
    let graph_pin = manifest
        .files
        .get(name)
        .ok_or_else(|| integrity(&manifest_path, format!("missing digest for graph `{name}`")))?;
    let graph_path = local_file(&root, name)?;
    verify_file(&graph_path, graph_pin)?;
    let file = std::fs::File::open(&graph_path).map_err(|error| read_error(&graph_path, error))?;
    // SAFETY: exports are immutable read-only source files, like checkpoint
    // shards. Mapping avoids copying a multi-gigabyte inline-weight graph.
    let mapped =
        unsafe { memmap2::Mmap::map(&file) }.map_err(|error| read_error(&graph_path, error))?;
    let model =
        Model::decode(&mapped[..]).map_err(|error| integrity(&graph_path, error.to_string()))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| integrity(&graph_path, "ONNX model has no graph"))?;
    let mut locations = BTreeSet::new();
    graph
        .collect(&mut locations)
        .map_err(|message| integrity(&graph_path, message))?;
    for function in &model.functions {
        collect_nodes(&function.nodes, &mut locations)
            .map_err(|message| integrity(&graph_path, message))?;
        collect_attributes(&function.attributes, &mut locations)
            .map_err(|message| integrity(&graph_path, message))?;
    }
    for training in &model.training {
        for graph in [&training.initialization, &training.algorithm]
            .into_iter()
            .flatten()
        {
            graph
                .collect(&mut locations)
                .map_err(|message| integrity(&graph_path, message))?;
        }
    }
    for location in locations {
        if !manifest.files.contains_key(&location) {
            return Err(integrity(
                &manifest_path,
                format!("external weight `{location}` has no digest"),
            ));
        }
    }
    for (name, pin) in &manifest.files {
        if name
            != model_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        {
            verify_file(&local_file(&root, name)?, pin)?;
        }
    }
    Ok(graph_pin.sha256.to_ascii_lowercase())
}

// Partial definitions of the ONNX protobuf schema retain only containers
// that can reference external tensors. Unknown fields (including inline
// tensor bytes) are skipped by prost. Field tags follow onnx/onnx.proto3.
#[derive(Clone, PartialEq, Message)]
struct Model {
    #[prost(message, optional, tag = "7")]
    graph: Option<Graph>,
    #[prost(message, repeated, tag = "20")]
    training: Vec<Training>,
    #[prost(message, repeated, tag = "25")]
    functions: Vec<Function>,
}

#[derive(Clone, PartialEq, Message)]
struct Training {
    #[prost(message, optional, tag = "1")]
    initialization: Option<Graph>,
    #[prost(message, optional, tag = "2")]
    algorithm: Option<Graph>,
}

#[derive(Clone, PartialEq, Message)]
struct Function {
    #[prost(message, repeated, tag = "7")]
    nodes: Vec<Node>,
    #[prost(message, repeated, tag = "11")]
    attributes: Vec<Attribute>,
}

#[derive(Clone, PartialEq, Message)]
struct Graph {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<Node>,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<Tensor>,
    #[prost(message, repeated, tag = "15")]
    sparse_initializers: Vec<SparseTensor>,
}

#[derive(Clone, PartialEq, Message)]
struct Node {
    #[prost(message, repeated, tag = "5")]
    attributes: Vec<Attribute>,
}

#[derive(Clone, PartialEq, Message)]
struct Attribute {
    #[prost(message, optional, tag = "5")]
    tensor: Option<Tensor>,
    #[prost(message, optional, tag = "6")]
    graph: Option<Graph>,
    #[prost(message, repeated, tag = "10")]
    tensors: Vec<Tensor>,
    #[prost(message, repeated, tag = "11")]
    graphs: Vec<Graph>,
    #[prost(message, optional, tag = "22")]
    sparse_tensor: Option<SparseTensor>,
    #[prost(message, repeated, tag = "23")]
    sparse_tensors: Vec<SparseTensor>,
}

#[derive(Clone, PartialEq, Message)]
struct SparseTensor {
    #[prost(message, optional, tag = "1")]
    values: Option<Tensor>,
    #[prost(message, optional, tag = "2")]
    indices: Option<Tensor>,
}

#[derive(Clone, PartialEq, Message)]
struct Tensor {
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<Entry>,
    #[prost(int32, tag = "14")]
    data_location: i32,
}

#[derive(Clone, PartialEq, Message)]
struct Entry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

impl Graph {
    fn collect(&self, locations: &mut BTreeSet<String>) -> Result<(), String> {
        collect_nodes(&self.nodes, locations)?;
        for tensor in &self.initializers {
            tensor.collect(locations)?;
        }
        for tensor in &self.sparse_initializers {
            tensor.collect(locations)?;
        }
        Ok(())
    }
}

fn collect_nodes(nodes: &[Node], locations: &mut BTreeSet<String>) -> Result<(), String> {
    for node in nodes {
        collect_attributes(&node.attributes, locations)?;
    }
    Ok(())
}

fn collect_attributes(
    attributes: &[Attribute],
    locations: &mut BTreeSet<String>,
) -> Result<(), String> {
    for attribute in attributes {
        for tensor in attribute.tensor.iter().chain(&attribute.tensors) {
            tensor.collect(locations)?;
        }
        for tensor in attribute
            .sparse_tensor
            .iter()
            .chain(&attribute.sparse_tensors)
        {
            tensor.collect(locations)?;
        }
        for graph in attribute.graph.iter().chain(&attribute.graphs) {
            graph.collect(locations)?;
        }
    }
    Ok(())
}

impl SparseTensor {
    fn collect(&self, locations: &mut BTreeSet<String>) -> Result<(), String> {
        for tensor in self.values.iter().chain(&self.indices) {
            tensor.collect(locations)?;
        }
        Ok(())
    }
}

impl Tensor {
    fn collect(&self, locations: &mut BTreeSet<String>) -> Result<(), String> {
        if self.data_location == 0 && self.external_data.is_empty() {
            return Ok(());
        }
        if self.data_location != 1 {
            return Err("invalid external tensor data location".to_owned());
        }
        let entries: Vec<_> = self
            .external_data
            .iter()
            .filter(|entry| entry.key == "location")
            .collect();
        if entries.len() != 1 {
            return Err("external tensor requires exactly one location".to_owned());
        }
        locations.insert(entries[0].value.clone());
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn write_test_manifest(model_path: &Path, external_files: &[&str]) {
    let mut files = BTreeMap::new();
    let root = model_path.parent().unwrap();
    for name in std::iter::once(model_path.file_name().unwrap().to_str().unwrap())
        .chain(external_files.iter().copied())
    {
        let bytes = std::fs::read(root.join(name)).unwrap();
        files.insert(
            name.to_owned(),
            FilePin {
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            },
        );
    }
    std::fs::write(
        manifest_path(model_path),
        serde_json::to_vec(&Manifest {
            schema_version: 1,
            files,
        })
        .unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_and_external_weights_must_match_manifest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        let weights = directory.path().join("weights.bin");
        let tensor = Tensor {
            external_data: vec![Entry {
                key: "location".into(),
                value: "weights.bin".into(),
            }],
            data_location: 1,
        };
        // Nest external weights in a graph attribute to exercise traversal
        // beyond top-level initializers.
        let model = Model {
            graph: Some(Graph {
                nodes: vec![Node {
                    attributes: vec![Attribute {
                        graph: Some(Graph {
                            initializers: vec![tensor],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }],
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        std::fs::write(&path, model.encode_to_vec()).unwrap();
        std::fs::write(&weights, b"weights").unwrap();
        assert!(matches!(verify(&path), Err(OnnxError::Io { .. })));
        write_test_manifest(&path, &[]);
        assert!(matches!(verify(&path), Err(OnnxError::Integrity { .. })));
        write_test_manifest(&path, &["weights.bin"]);
        verify(&path).unwrap();
        std::fs::write(&weights, b"changed").unwrap();
        assert!(matches!(verify(&path), Err(OnnxError::Integrity { .. })));
        write_test_manifest(&path, &["weights.bin"]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(verify(&path), Err(OnnxError::Integrity { .. })));
    }

    #[test]
    fn manifest_paths_cannot_escape_model_root() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        for name in ["../weights", "/weights", "", "..\\weights"] {
            assert!(matches!(
                local_file(&root, name),
                Err(OnnxError::Integrity { .. })
            ));
        }
        #[cfg(unix)]
        {
            let outside = tempfile::NamedTempFile::new().unwrap();
            std::os::unix::fs::symlink(outside.path(), root.join("weights")).unwrap();
            assert!(matches!(
                local_file(&root, "weights"),
                Err(OnnxError::Integrity { .. })
            ));
        }
    }
}
