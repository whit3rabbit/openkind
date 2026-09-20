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
  -h, --help              Show this help.

The script refuses a dirty tree so every result has one unambiguous subject SHA.
It never downloads model artifacts.
EOF
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "error: run inside the opendecision Git repository" >&2
  exit 2
fi
cd "${repo_root}"

checkpoint_root=""
reference_root="research/OpenDecision_Phase3B_BackboneParity_20260920T152206Z"
head_bundle_root="crates/opendecision-backends/tests/fixtures/qwen35_statefirst_a047d6802c3f06f085b8"
output_dir=""
scheduler_reps=""
offline_only=0

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
  echo "==> ${name}"
  "$@" 2>&1 | tee "${output_dir}/${name}.log"
}

count_tests() {
  local log="$1"
  awk '/: test$/ { count += 1 } END { print count + 0 }' "${log}"
}

{
  echo "subject_sha=${subject_sha}"
  echo "subject_short=${subject_short}"
  echo "started_utc=${started_utc}"
  echo "branch=$(git branch --show-current)"
  echo "offline_only=${offline_only}"
  echo "reference_root=${reference_root}"
  echo "head_bundle_root=${head_bundle_root}"
  if [[ "${offline_only}" -eq 0 ]]; then
    echo "checkpoint_directory_name=$(basename "${checkpoint_root}")"
  fi
} > "${output_dir}/manifest.env"

run_gate git-status-before git status --short
run_gate rustc-version rustc -Vv
run_gate macos-version sw_vers
run_gate hardware-model sysctl -n hw.model
run_gate hardware-memory sysctl -n hw.memsize
run_gate cargo-fmt cargo fmt --check
run_gate cargo-clippy cargo clippy --workspace --all-targets --all-features -- -D warnings
run_gate cargo-test-default env -u RUST_LOG cargo test --workspace
run_gate cargo-test-default-list env -u RUST_LOG cargo test --workspace -- --list
default_test_count="$(count_tests "${output_dir}/cargo-test-default-list.log")"
echo "default_test_count=${default_test_count}" >> "${output_dir}/manifest.env"
run_gate cargo-test-all-features env -u RUST_LOG cargo test --workspace --all-features
run_gate cargo-test-all-features-list env -u RUST_LOG cargo test --workspace --all-features -- --list
all_features_test_count="$(count_tests "${output_dir}/cargo-test-all-features-list.log")"
echo "all_features_test_count=${all_features_test_count}" >> "${output_dir}/manifest.env"

run_gate schema-regeneration cargo run -p opendecision-gen-schemas -- --write
if ! git diff --quiet -- crates/opendecision-core/schemas; then
  echo "error: schema regeneration changed committed schema files" >&2
  git diff -- crates/opendecision-core/schemas >&2
  exit 1
fi
if [[ -n "$(git status --porcelain -- crates/opendecision-core/schemas)" ]]; then
  echo "error: schema regeneration created untracked schema files" >&2
  git status --short -- crates/opendecision-core/schemas >&2
  exit 1
fi
echo "schema_zero_diff=true" >> "${output_dir}/manifest.env"

run_gate scheduler-stress cargo run -p opendecision-backends --example qwen35_scheduler_stress

if [[ "${offline_only}" -eq 0 ]]; then
  run_gate checkpoint-digests shasum -a 256 \
    "${checkpoint_root}/model.safetensors-00001-of-00002.safetensors" \
    "${checkpoint_root}/model.safetensors-00002-of-00002.safetensors"
  run_gate qwen35-full-parity cargo run --release -p opendecision-backends \
    --example qwen35_full_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  run_gate qwen35-branch-parity cargo run --release -p opendecision-backends \
    --example qwen35_parity_probe -- \
    "${checkpoint_root}" "${reference_root}" branch
  run_gate qwen35-nested-parity cargo run --release -p opendecision-backends \
    --example qwen35_nested_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  run_gate qwen35-batched-parity cargo run --release -p opendecision-backends \
    --example qwen35_batched_parity -- \
    "${checkpoint_root}" "${reference_root}" "${head_bundle_root}"
  if [[ -n "${scheduler_reps}" ]]; then
    run_gate qwen35-scheduler-benchmark cargo run --release -p opendecision-backends \
      --example qwen35_scheduler_bench -- \
      "${checkpoint_root}" "${reference_root}" "${scheduler_reps}"
  else
    run_gate qwen35-scheduler-benchmark cargo run --release -p opendecision-backends \
      --example qwen35_scheduler_bench -- \
      "${checkpoint_root}" "${reference_root}"
  fi
fi

run_gate subject-diff-check git diff --check "${subject_sha}^" "${subject_sha}"
run_gate working-tree-diff-check git diff --check
run_gate git-status-after git status --short
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "error: verification changed the working tree" >&2
  git status --short >&2
  exit 1
fi

finished_utc="$(date -u +%Y%m%dT%H%M%SZ)"
echo "finished_utc=${finished_utc}" >> "${output_dir}/manifest.env"
echo "verification passed for ${subject_sha}"
echo "logs: ${output_dir}"
