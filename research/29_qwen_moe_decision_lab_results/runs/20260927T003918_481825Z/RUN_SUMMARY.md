# Follow-up run summary

**Status:** EXPLORATORY_COMPLETE
**Model:** Qwen/Qwen1.5-MoE-A2.7B-Chat at ec052fda178e241c7c443468d2fa1db6618996be
**Selected prompt:** explicit_three_way
**Development-selected candidate:** skip_last_6
**Research gate:** False

All quality/timing results use full-input execution. Cache diagnostics are separate.

| Arm | Accuracy | Unknown recall | Coverage | Accepted error | p50 ms |
|---|---:|---:|---:|---:|---:|
| native | 0.375 | 0.000 | 0.146 | 0.643 | 2022.5 |
| skip_last_6 | 0.417 | 0.062 | 0.083 | 0.375 | 1535.8 |

## Limits
- Fresh data are authored fixtures, not independent production/benchmark evidence
- Historical Drive run is incomplete; the later pasted physical-removal result lacks raw Drive artifacts
- NF4 FP32 linear reference retains NF4 weights and lower-precision surrounding operations
- No actual expert streaming, distillation, weight pruning, or Qwen3.5 hybrid-cache execution
- Hook/eager runtime is not a grouped-expert serving speed ceiling
