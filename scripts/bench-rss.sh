#!/usr/bin/env bash

# Measure one Criterion benchmark filter in its own process so Cargo's build
# process is not confused with the benchmark's resident-memory high-water mark.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/bench-rss.sh --package PACKAGE --bench TARGET --filter BENCHMARK_ID [options]

Options:
  --package PACKAGE    Cargo package containing the Criterion target.
  --bench TARGET       Exact Cargo bench target name.
  --filter ID          Exact Criterion benchmark id to profile.
  --duration SECONDS   Criterion profile duration (default: 5).
  --output PATH        Log path (default: target/bench-rss/<SHA>/...).
  -h, --help           Show this help.

Supported hosts: macOS (/usr/bin/time -l) and Linux (/usr/bin/time -v).
The log records provenance and a process-level peak RSS. It is not bytes per
operation and should only be compared on the same host and toolchain.
EOF
}

package=""
bench=""
filter=""
duration="5"
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package)
      package="${2:?--package requires a value}"
      shift 2
      ;;
    --bench)
      bench="${2:?--bench requires a value}"
      shift 2
      ;;
    --filter)
      filter="${2:?--filter requires a value}"
      shift 2
      ;;
    --duration)
      duration="${2:?--duration requires a value}"
      shift 2
      ;;
    --output)
      output="${2:?--output requires a path}"
      shift 2
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

if [[ -z "${package}" || -z "${bench}" || -z "${filter}" ]]; then
  echo "error: --package, --bench, and --filter are required" >&2
  usage >&2
  exit 2
fi
if [[ ! "${duration}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]]; then
  echo "error: --duration must be a positive number of seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "error: run inside the openkind Git repository" >&2
  exit 2
fi
cd "${repo_root}"

case "$(uname -s)" in
  Darwin)
    time_args=(-l)
    ;;
  Linux)
    time_args=(-v)
    ;;
  *)
    echo "error: peak-RSS measurement supports only macOS and Linux" >&2
    exit 2
    ;;
esac
if [[ ! -x /usr/bin/time ]]; then
  echo "error: /usr/bin/time is required" >&2
  exit 2
fi

artifact_log="$(mktemp "${TMPDIR:-/tmp}/openkind-bench-rss.XXXXXX")"
trap 'rm -f "${artifact_log}"' EXIT

cargo bench --locked -p "${package}" --bench "${bench}" --no-run \
  --message-format=json > "${artifact_log}"

bench_executable="$({
  awk -v target="\"name\":\"${bench}\"" '
    index($0, "\"reason\":\"compiler-artifact\"") &&
    index($0, "\"kind\":[\"bench\"]") &&
    index($0, target) { print }
  ' "${artifact_log}"
} | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -n 1)"

if [[ -z "${bench_executable}" || ! -x "${bench_executable}" ]]; then
  echo "error: Cargo did not report an executable for bench target ${bench}" >&2
  exit 1
fi

subject_sha="$(git rev-parse HEAD)"
subject_short="$(git rev-parse --short=12 HEAD)"
dirty=false
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  dirty=true
fi
safe_filter="$(printf '%s' "${filter}" | tr -cs 'A-Za-z0-9._-' '_')"
criterion_filter="^$(printf '%s' "${filter}" | sed 's/[][(){}.^$*+?|\\]/\\&/g')$"

if [[ -z "${output}" ]]; then
  output="target/bench-rss/${subject_short}/${package}-${bench}-${safe_filter}.log"
fi
mkdir -p "$(dirname "${output}")"

{
  echo "subject_sha=${subject_sha}"
  echo "working_tree_dirty=${dirty}"
  echo "package=${package}"
  echo "bench=${bench}"
  echo "filter=${filter}"
  echo "profile_seconds=${duration}"
  echo "executable=${bench_executable}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -a)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "command=${bench_executable} --bench ${criterion_filter} --profile-time ${duration} --noplot"
} > "${output}"

{
  /usr/bin/time "${time_args[@]}" "${bench_executable}" \
    --bench "${criterion_filter}" --profile-time "${duration}" --noplot
} 2>&1 | tee -a "${output}"

echo "rss log: ${output}"
