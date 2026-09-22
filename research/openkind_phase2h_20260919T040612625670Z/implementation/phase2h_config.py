"""Phase 2H preregistration. Execution, learning, and evaluation remain separate."""
from dataclasses import dataclass,asdict
from pathlib import Path
VERSION='2h.1.2'
G_SHA='24ff15fa1fec3c8eeb17ef073da3d852cabe40683678b247d4e1d917e847414b'
G_RUN='20260919T005142584348Z'
@dataclass
class Settings:
    preset:str='standard'
    reference_archive:str='/content/drive/MyDrive/Colab Notebooks/OpenKind_Phase2G_results/20260919T005142584348Z/openkind_phase2g_20260919T005142584348Z.zip'
    output_root:str='/content/openkind_phase2h'
    cache_root:str='/content/openkind_phase2h_cache'
    drive_destination:str='/content/drive/MyDrive/Colab Notebooks/OpenKind_Phase2H_results'
    registry_path:str='/content/drive/MyDrive/Colab Notebooks/OpenKind_Phase2H_registry/reserved_groups.json'
    copy_results_to_drive:bool=True
    seed:int=20260923
    head_seeds:tuple=(17,29,43)
    candidate_counts:tuple=(4,16)
    fit_candidate_counts:tuple=(4,8)
    max_length:int=512
    context_max_length:int=1792
    context_min_tokens:tuple=(0,256,1024)
    head_epochs:int=30
    head_patience:int=5
    head_lr:float=0.0003
    head_weight_decay:float=0.001
    head_batch_size:int=32
    none_l2:float=0.001
    train_mixture:tuple=(0.60,0.25,0.15) # present / omitted / author OOS; assumptions, not measured traffic
    calibration_min_improvement:float=0.002
    wrong_answer_costs:tuple=(1.,5.,20.)
    absent_priors:tuple=(0.05,0.25,0.5)
    oos_share_of_absent:float=0.5
    review_cost:float=0.1
    bootstrap_repeats:int=500
    probability_tolerance:float=0.005
    benchmark_repeats:int=3
    checkpoint_every:int=32
    run_final:bool=True
    run_tf32:bool=True
    run_robustness:bool=True
    run_finite_token:bool=True
    run_primitives:bool=True
    run_state_first:bool=False
    run_smaller_model:bool=False
    run_lora:bool=False
    smaller_model_id:str='Qwen/Qwen3.5-2B-Base'
    criteria_review_json:str=''
    require_independent_review_for_final:bool=False
    natural_documents_jsonl:str=''
    lora_rank:int=8
    lora_alpha:int=16
    lora_last_blocks:int=4
    lora_epochs:int=2
    lora_pairs_cap:int=1024
    lora_lr:float=0.00005
    lora_accumulation:int=16
    lora_min_gpu_gib:float=32.0
    primitive_max_length:int=1024
    def sizes(self):
        sizes={
            'smoke':dict(train=4,dev=4,cal_fit=4,cal_gate=4,policy_dev=4,final=4,oos_train=4,oos_aux=2,oos_final=4,criteria_per_label=1,robust_per_family=1,bench=2,primitive_train=8,primitive_aux=4,primitive_final=8),
            'quick':dict(train=48,dev=16,cal_fit=16,cal_gate=16,policy_dev=16,final=16,oos_train=12,oos_aux=6,oos_final=16,criteria_per_label=1,robust_per_family=2,bench=4,primitive_train=64,primitive_aux=16,primitive_final=32),
            'standard':dict(train=256,dev=64,cal_fit=64,cal_gate=48,policy_dev=64,final=64,oos_train=40,oos_aux=12,oos_final=64,criteria_per_label=2,robust_per_family=4,bench=8,primitive_train=384,primitive_aux=64,primitive_final=128)}
        if self.preset not in sizes:raise ValueError('preset must be smoke, quick, or standard')
        if not self.head_seeds or len(set(self.head_seeds))!=len(self.head_seeds):raise ValueError('Use distinct head seeds')
        if abs(sum(self.train_mixture)-1)>1e-9 or min(self.train_mixture)<=0:raise ValueError('Invalid training population mixture')
        if self.max_length<256 or self.max_length>1024:raise ValueError('max_length outside study range')
        if not 0<self.probability_tolerance<=0.005:raise ValueError('Do not relax the historical gate')
        if self.lora_min_gpu_gib<24:raise ValueError('Do not bypass LoRA training memory guard')
        return sizes[self.preset]
    def json(self):return asdict(self)
