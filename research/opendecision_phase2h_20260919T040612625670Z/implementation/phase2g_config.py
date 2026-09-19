"""Phase 2G: fresh task evidence and explicitly versioned serving experiments."""
from dataclasses import dataclass, asdict
from pathlib import Path

VERSION = '2g.1.0'
F_RUN = '20260918T224427722898Z'
F_SHA = 'e1d3fea878d35b9e929e83424d4f3aa8b5de0385bb100ace001110f7e5f83c37'
EXCLUSION_SHA = 'f3549213c79dc4530dd8d04ecf61e5a6f08b8ba3e470a3a00d8296bc7bc5b42a'
CLINC_COMMIT = '828f8093932c8fe6ca7936c3d2e52903b1c523de'
CLINC_BLOBS = {'data/data_full.json': '7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa',
                'data/domains.json': '60a74358e52060e60b6128ae4efe679e9d6ba69c',
                'LICENSE': '1a16e05564d2aaa880bbe9e506a0a0226d8742cc'}
BANK_TEST_SHA = 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d'

@dataclass
class Settings:
    preset: str = 'standard'
    reference_archive: str = '/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2F_results/20260918T224427722898Z/opendecision_phase2f_20260918T224427722898Z.zip'
    expected_reference_sha256: str = F_SHA
    output_root: str = '/content/opendecision_phase2g'
    cache_root: str = '/content/opendecision_phase2g_cache'
    drive_destination: str = '/content/drive/MyDrive/Colab Notebooks/OpenDecision_Phase2G_results'
    model_id: str = 'Qwen/Qwen3.5-4B-Base'
    model_revision: str = '1001bb4d826a52d1f399e183466143f4da7b741b'
    modes: tuple = ('fp32_strict_math', 'fp32_tf32_allowed')
    # New mode retains FP32 weights/activations and permits TF32 CUDA GEMM; no promised speed/parity.
    max_length: int = 256
    candidate_counts: tuple = (4,16)
    seed: int = 20260919
    suffix_batch_size: int = 4
    probability_tolerance: float = .005
    cache_probability_tolerance: float = .005
    order_probability_tolerance: float = 1e-6
    quantization_seed: int = 42  # inherited cache identity field; NO low-bit codec is executed
    fp32_workspace_gib: float = 2.0
    initial_gpu_allocation_limit_mib: float = 64.0
    run_fresh_quality: bool = True
    run_cache_parity: bool = True
    run_labeled_context: bool = True
    run_traffic: bool = True
    run_request_benchmark: bool = True
    run_crossdomain: bool = True
    copy_results_to_drive: bool = True
    benchmark_repeats: int = 3
    benchmark_warmups: int = 1
    bootstrap_repeats: int = 500
    traffic_repeats: int = 2
    prefix_cache_mib: int = 192
    prefix_cache_ttl_seconds: float = 20.0
    context_min_tokens: tuple = (0,256,1024)
    context_max_length: int = 1536
    custom_jsonl: str = ''  # optional separately reported labeled source; no auto-generated labels
    checkpoint_every: int = 8
    require_torch_series: str = '2.11.'

    def sizes(self):
        presets = {
          'smoke': dict(messages_per_family=2, oos_messages=2, parity_per_family=1,
                        benchmark_messages=2, context_per_family=1, traffic_messages=3, traffic_requests=12),
          'quick': dict(messages_per_family=8, oos_messages=4, parity_per_family=2,
                        benchmark_messages=4, context_per_family=1, traffic_messages=6, traffic_requests=24),
          'standard': dict(messages_per_family=32, oos_messages=16, parity_per_family=4,
                        benchmark_messages=8, context_per_family=2, traffic_messages=12, traffic_requests=48)}
        if self.preset not in presets: raise ValueError('Use smoke, quick or standard.')
        if self.expected_reference_sha256 != F_SHA: raise ValueError('Reference must be the completed Phase 2F archive.')
        if not self.modes or len(set(self.modes))!=len(self.modes) or set(self.modes)-{'fp32_strict_math','fp32_tf32_allowed','bf16_default'}:
            raise ValueError('Invalid execution modes.')
        if self.modes[0]!='fp32_strict_math': raise ValueError('Run the FP32 reference first.')
        if self.max_length!=256 or tuple(self.candidate_counts)!=(4,16): raise ValueError('This registered design uses short limit 256 and K=4,16.')
        if not 0<self.cache_probability_tolerance<=.005: raise ValueError('Do not relax the prior numerical gate.')
        if not 32<=self.prefix_cache_mib<=1024 or self.prefix_cache_ttl_seconds<=0: raise ValueError('Invalid cache budget/TTL.')
        for k in ('benchmark_repeats','traffic_repeats','bootstrap_repeats','checkpoint_every'):
            if getattr(self,k)<1: raise ValueError(k+' must be positive')
        if not 256<=self.context_max_length<=2048 or any(x<0 or x>=self.context_max_length-128 for x in self.context_min_tokens):
            raise ValueError('Invalid context-length controls.')
        return presets[self.preset]

    def json(self): return asdict(self)
