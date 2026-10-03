#!/usr/bin/env bash

# Verify one clean commit and retain complete logs under target/verification.
# Model-backed gates are required unless --offline-only is explicit.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify-commit.sh --checkpoint-root PATH [options]
  scripts/verify-commit.sh --offline-only [options]

Options:
  --checkpoint-root PATH  Pinned Qwen3.5-4B checkpoint directory.
  --reference-root PATH   Phase 3B reference root.
  --head-bundle-root PATH Selected head/golden-fixture bundle root.
  --output-dir PATH       Log directory (default: target/verification/<SHA>-<UTC>).
  --scheduler-reps N      Override scheduler samples. Omit for the canonical default.
  --offline-only          Skip checkpoint-backed parity and benchmark gates.
  --cuda                  Include native CUDA checks on a CUDA-toolchain host.
  -h, --help              Show this help.

The script refuses a dirty tree so every result has one unambiguous subject SHA.
It never downloads model artifacts.
EOF
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "error: run inside the openkind Git repository" >&2
  exit 2
fi
cd "${repo_root}"

checkpoint_root=""
reference_root="research/14_phase3b_backbone_parity_results"
head_bundle_root="crates/openkind-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8"
output_dir=""
scheduler_reps=""
offline_only=0
cuda_requested=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root)
      checkpoint_root="${2:?--checkpoint-root requires a path}"
      shift 2
      ;;
    --reference-root)
      reference_root="${2:?--reference-root requires a path}"
      shift 2
      ;;
    --head-bundle-root)
      head_bundle_root="${2:?--head-bundle-root requires a path}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a path}"
      shift 2
      ;;
    --scheduler-reps)
      scheduler_reps="${2:?--scheduler-reps requires a positive integer}"
      shift 2
      ;;
    --offline-only)
      offline_only=1
      shift
      ;;
    --cuda)
      cuda_requested=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${scheduler_reps}" && ! "${scheduler_reps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --scheduler-reps must be a positive integer" >&2
  exit 2
fi
if [[ "${offline_only}" -eq 0 && -z "${checkpoint_root}" ]]; then
  echo "error: --checkpoint-root is required unless --offline-only is explicit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "error: verification requires a clean working tree" >&2
  git status --short >&2
  exit 2
fi
if [[ ! -d "${reference_root}" ]]; then
  echo "error: reference root not found: ${reference_root}" >&2
  exit 2
fi
if [[ ! -d "${head_bundle_root}" ]]; then
  echo "error: head bundle root not found: ${head_bundle_root}" >&2
  exit 2
fi
if [[ "${offline_only}" -eq 0 && ! -d "${checkpoint_root}" ]]; then
  echo "error: checkpoint root not found: ${checkpoint_root}" >&2
  exit 2
fi

# ONNX providers are dynamically loaded; native CUDA needs an explicit toolchain gate.
host_os="$(uname -s)"
host_arch="$(uname -m)"
host_features="onnx,onnx-cuda,onnx-rocm"
if [[ "${host_os}" == Darwin && "${host_arch}" == arm64 ]]; then
  host_features="mlx,${host_features}"
  SDKROOT="$(xcrun --show-sdk-path)"
  export SDKROOT
fi
if [[ "${cuda_requested}" -eq 1 ]]; then
  if [[ "${host_os}" == Darwin ]] || ! command -v nvcc > /dev/null; then
    echo "error: --cuda requires a supported Linux/Windows host with nvcc" >&2
    exit 2
  fi
  host_features="${host_features},cuda"
fi

subject_sha="$(git rev-parse HEAD)"
subject_short="$(git rev-parse --short=12 HEAD)"
started_utc="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${output_dir}" ]]; then
  output_dir="target/verification/${subject_short}-${started_utc}"
fi
mkdir -p "${output_dir}"

run_gate() {
  local name="$1"
  shift
  current_gate="$name"
  echo "==> ${name}"
  "$@" 2>&1 | tee "${output_dir}/${name}.log"
}

count_tests() {
  local log="$1"
  awk '/: test$/ { count += 1 } END { print count + 0 }' "${log}"
}

current_gate=initialization
finish_report() {
  local result=$?
  echo "finished_utc=$(date -u +%Y%m%dT%H%M%SZ)" >> "${output_dir}/manifest.env"
  echo "exit_status=${result}" >> "${output_dir}/manifest.env"
  if [[ "${result}" -ne 0 ]]; then
    echo "failed_gate=${current_gate}" >> "${output_dir}/manifest.env"
    echo "verification failed during ${current_gate}; logs: ${output_dir}" >&2
  fi
}
trap finish_report EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  echo "subject_sha=${subject_sha}"
  echo "subject_short=${subject_short}"
  echo "started_utc=${started_utc}"
  echo "branch=$(git branch --show-current)"
  echo "offline_only=${offline_only}"
  echo "host_os=${host_os}"
  echo "host_arch=${host_arch}"
  echo "host_features=${host_features}"
  echo "cuda_requested=${cuda_requested}"
  echo "reference_root=${reference_root}"
  echo "head_bundle_root=${head_bundle_root}"
  if [[ "${offline_only}" -eq 0 ]]; then
    echo "checkpoint_directory_name=$(basename "${checkpoint_root}")"
  fi
} > "${output_dir}/manifest.env"

run_gate git-status-before git status --short
run_gate rustc-version rustc -Vv
if [[ "${host_os}" == Darwin ]]; then
  run_gate macos-version sw_vers
  run_gate hardware-model sysctl -n hw.model
  run_gate hardware-memory sysctl -n hw.memsize
else
  run_gate host-version uname -a
fi
run_gate available-disk df -h
run_gate cargo-fmt cargo fmt --check
run_gate cargo-clippy-default cargo clippy --workspace --all-targets --locked -- -D warnings
run_gate cargo-clippy-host cargo clippy --workspace --all-targets --features "${host_features}" --locked -- -D warnings
run_gate cargo-test-default env -u RUST_LOG cargo test --workspace --locked
run_gate cargo-test-default-list env -u RUST_LOG cargo test --workspace --locked -- --list
default_test_count="$(count_tests "${output_dir}/cargo-test-default-list.log")"
echo "default_test_count=${default_test_count}" >> "${output_dir}/manifest.env"
run_gate cargo-test-host env -u RUST_LOG cargo test --workspace --features "${host_features}" --locked
run_gate cargo-test-host-list env -u RUST_LOG cargo test --workspace --features "${host_features}" --locked -- --list
host_features_test_count="$(count_tests "${output_dir}/cargo-test-host-list.log")"
echo "host_features_test_count=${host_features_test_count}" >> "${output_dir}/manifest.env"
for feature in onnx onnx-cuda onnx-rocm; do
  run_gate "cargo-check-backends-${feature}" cargo check -p openkind-backends --all-targets --features "${feature}" --locked
done
run_gate cargo-test-benches env -u RUST_LOG cargo test --workspace --benches --locked

run_gate schema-regeneration cargo run -p openkind-gen-schemas --locked -- --write
if ! git diff --quiet -- crates/openkind-core/schemas; then
  echo "error: schema regeneration changed committed schema files" >&2
  git diff -- crates/openkind-core/schemas >&2
  exit 1
fi
if [[ -n "$(git status --porcelain -- crates/openkind-core/schemas)" ]]; then
  echo "error: schema regeneration created untracked schema files" >&2
  git status --short -- crates/openkind-core/schemas >&2
  exit 1
fi
echo "schema_zero_diff=true" >> "${output_dir}/manifest.env"

run_gate scheduler-stress cargo run -p openkind-backends --locked --example qwen35_scheduler_stress

if [[ "${offline_only}" -eq 0 ]]; then
  run_gate checkpoint-digests shasum -a 256 \
    "${checkpoint_root}/model.safetensors-00001-of-00002.safetensors" \
    "${checkpoint_root}/model.safetensors-00002-of-00002.safetensors"
  run_gate qwen35-full-parity cargo run --release --locked -p openkind-backends \
    --example qwen35_full_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  run_gate qwen35-branch-parity cargo run --release --locked -p openkind-backends \
    --example qwen35_parity_probe -- \
    "${checkpoint_root}" "${reference_root}" branch
  run_gate qwen35-nested-parity cargo run --release --locked -p openkind-backends \
    --example qwen35_nested_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  run_gate qwen35-batched-parity cargo run --release --locked -p openkind-backends \
    --example qwen35_batched_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  if [[ -n "${scheduler_reps}" ]]; then
    run_gate qwen35-scheduler-benchmark cargo run --release --locked -p openkind-backends \
      --example qwen35_scheduler_bench -- \
      "${checkpoint_root}" "${reference_root}" "${scheduler_reps}"
  else
    run_gate qwen35-scheduler-benchmark cargo run --release --locked -p openkind-backends \
      --example qwen35_scheduler_bench -- \
      "${checkpoint_root}" "${reference_root}"
  fi
fi

run_gate subject-diff-check git diff --check "${subject_sha}^" "${subject_sha}"
run_gate working-tree-diff-check git diff --check
run_gate git-status-after git status --short
current_gate=working-tree-cleanliness
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "error: verification changed the working tree" >&2
  git status --short >&2
  exit 1
fi

echo "verification passed for ${subject_sha}"
echo "logs: ${output_dir}"
