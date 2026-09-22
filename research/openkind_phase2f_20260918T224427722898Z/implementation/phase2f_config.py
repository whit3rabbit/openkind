"""Phase 2F: frozen decision graph; cache execution and storage experiments only."""
from dataclasses import dataclass, asdict
from pathlib import Path

VERSION = '2f.1.0'
BASELINE_RUN = '20260918T114914072764Z'
BASELINE_SHA256 = '7154fd3e164a1fbe71ded6ced3ae903bab7d2870dc9188748e7343d44221604b'
TURBOQUANT_COMMIT = '31660314b229b8d1c29bfddf8b9f5026b5121095'

@dataclass
class Settings:
    preset: str = 'standard'  # smoke / quick / standard
    reference_archive: str = '/content/drive/MyDrive/Colab Notebooks/OpenKind_Phase2E_expanded_results/20260918T114914072764Z/openkind_phase2e_expanded_20260918T114914072764Z.zip'
    expected_reference_sha256: str = BASELINE_SHA256
    output_root: str = '/content/openkind_phase2f'
    cache_root: str = '/content/openkind_phase2f_cache'
    drive_destination: str = '/content/drive/MyDrive/Colab Notebooks/OpenKind_Phase2F_results'
    model_id: str = 'Qwen/Qwen3.5-4B-Base'
    model_revision: str = '1001bb4d826a52d1f399e183466143f4da7b741b'
    modes: tuple = ('fp32_strict_math', 'bf16_default')
    max_length: int = 256
    suffix_batch_size: int = 4
    probability_tolerance: float = .005
    cache_probability_tolerance: float = .005
    order_probability_tolerance: float = 1e-6
    run_turboquant: bool = True
    run_cache_compression: bool = True
    run_request_benchmarks: bool = True
    run_cross_request_cache: bool = True
    run_long_prefix: bool = True
    run_cpu_offload: bool = True
    copy_results_to_drive: bool = True
    codec_specs: tuple = ('lossless', 'kv_fp16', 'tq_k3_v4_r0', 'tq_k3_v4_r32', 'tq_k4_v4_r32', 'tq_k3_v2_r32')
    quantization_seed: int = 42
    # Read-only cache entries are byte-budgeted; model, transient branches and codec tables are separate.
    prefix_cache_mib: int = 384
    prefix_cache_ttl_seconds: float = 600.
    workload_rounds: int = 2
    benchmark_repeats: int = 5
    benchmark_warmups: int = 2
    profile_repeats: int = 2
    long_prefix_lengths: tuple = (64, 256, 1024)
    long_prefix_repeats: int = 3
    fp32_workspace_gib: float = 2.
    initial_gpu_allocation_limit_mib: float = 64.
    seed: int = 97
    turboquant_commit: str = TURBOQUANT_COMMIT

    def sizes(self):
        sizes = {
          'smoke': dict(groups_per_split=1, compression_per_stratum=1, benchmark_per_stratum=1),
          'quick': dict(groups_per_split=2, compression_per_stratum=1, benchmark_per_stratum=1),
          'standard': dict(groups_per_split=4, compression_per_stratum=2, benchmark_per_stratum=1)}
        if self.preset not in sizes: raise ValueError('preset must be smoke, quick or standard')
        if self.max_length != 256: raise ValueError('The frozen training prompt limit must remain 256.')
        if self.expected_reference_sha256 != BASELINE_SHA256: raise ValueError('Baseline must be the successful expanded E run, not a skipped T4 run.')
        if self.turboquant_commit != TURBOQUANT_COMMIT: raise ValueError('New upstream revisions require an explicit adapter review.')
        if not self.modes or len(set(self.modes)) != len(self.modes) or set(self.modes)-{'fp32_strict_math','bf16_default'}: raise ValueError('Invalid execution modes.')
        if self.suffix_batch_size != 4: raise ValueError('The registered baseline uses suffix batches of four; batch eight is a separate ablation.')
        allowed={'lossless','kv_fp16','tq_k3_v4_r0','tq_k3_v4_r32','tq_k4_v4_r32','tq_k3_v2_r32'}
        if not self.codec_specs or set(self.codec_specs)-allowed or self.codec_specs[0]!='lossless': raise ValueError('Invalid codec experiment list.')
        for n in ('benchmark_repeats','profile_repeats','long_prefix_repeats','workload_rounds'):
            if getattr(self,n)<1: raise ValueError(n+' must be positive')
        if self.benchmark_warmups<0 or not 32<=self.prefix_cache_mib<=1024 or self.prefix_cache_ttl_seconds<=0: raise ValueError('Invalid cache/timing limits.')
        if not 0<self.cache_probability_tolerance<=.005: raise ValueError('Do not relax the prior probability tolerance.')
        if any(not isinstance(n,int) or n<16 or n>4096 for n in self.long_prefix_lengths): raise ValueError('Synthetic prefixes must be 16..4096 tokens.')
        return sizes[self.preset]

    def json(self): return asdict(self)
