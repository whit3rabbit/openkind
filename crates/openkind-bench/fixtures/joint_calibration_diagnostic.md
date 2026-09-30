# Joint calibration partition

[`joint_calibration_diagnostic.jsonl`](joint_calibration_diagnostic.jsonl)
contains 64 authored Choice questions, all with explicit `__none__`
descriptions. Each of four tasks has four source groups with one example per
gold label (`alpha`, `beta`, `gamma`, `__none__`), giving 16 groups and 16
semantic-none cases.

| Task | Evidence and gold rule |
|---|---|
| `shift_route` | Match the stated filing time to the offered shift windows. |
| `amount_band` | Band the stated total; credit-denominated and missing totals are none. |
| `sensor_max` | Compare readings supplied in option descriptions; select the largest only if it is at least 120. |
| `posted_window` | Classify the stated posting date; missing dates are none. |

This panel is the fitting partition for the post-hoc calibration experiment
(`openkind-bench calibrate-choice`). It is disjoint from
[`joint_gate_diagnostic.jsonl`](joint_gate_diagnostic.jsonl) and from the
2026-09-29 joint-choice panels by task, record, and source group. The panels
were authored before any scoring run of this experiment.

This is a small, templated mechanism diagnostic, not a representative
natural-language benchmark, a promotion gate, or a historical calibration
partition. Its task and record names do not appear in any prior OpenKind
panel.
