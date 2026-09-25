"""Small CPU-only contracts executed by Colab before expensive work."""
import numpy as np
import torch
from phase2h_config import Settings
from phase2h_learning import AffineScorer,softmaxes,policy_action,weights,parity
from phase2h_extensions import LowRankLinear
from phase2e_runtime import indexing_selftest

def run_contract_checks():
    checks=[]
    report=indexing_selftest();checks.append('integer_tensor_indexing')
    p=softmaxes([np.array([1000.,1001.,999.])])[0]
    assert np.isfinite(p).all() and abs(p.sum()-1)<1e-12;checks.append('stable_probability_normalization')
    assert policy_action(np.array([.5,.1,.4]),.5)==0;assert policy_action(np.array([.4,.2,.4]),.3) is None;checks.append('threshold_and_none_tie_contract')
    torch.manual_seed(71);m=AffineScorer(6);x=torch.randn(12,6);before=m(x).detach().clone();m.fit_normalization(x);assert torch.allclose(m(x),before,atol=1e-6);checks.append('train_normalization_preserves_initial_affine')
    b=torch.nn.Linear(6,4);adapter=LowRankLinear(b,2,4);assert torch.equal(adapter(x),b(x));checks.append('zero_initialized_lora_is_identity')
    es=[{'group':'a','none_origin':'answer_present','choices':[{},{}],'target_index':0},{'group':'a','none_origin':'annotated_intent_omitted','choices':[{},{}],'target_index':2},{'group':'b','none_origin':'author_oos','choices':[{},{}],'target_index':2}]
    assert np.allclose(weights(es),[.60,.25,.15]);checks.append('named_population_mixture')
    assert parity([p]*3,[p]*3,es,.005,[])['accepted']==3;checks.append('unchanged_distribution_parity')
    return {'status':'passed','count':len(checks),'checks':checks,'scope':'tiny CPU contracts only, not Qwen performance, trained quality, independent review, or CUDA correctness','indexing':report}
