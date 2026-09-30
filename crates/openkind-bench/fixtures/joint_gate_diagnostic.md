# Joint gate partition

[`joint_gate_diagnostic.jsonl`](joint_gate_diagnostic.jsonl) contains 64
authored Choice questions, all with explicit `__none__` descriptions. Each of
four tasks has four source groups with one example per gold label (`alpha`,
`beta`, `gamma`, `__none__`), giving 16 groups and 16 semantic-none cases.

| Task | Evidence and gold rule |
|---|---|
| `dock_band` | Match the stated dock number to the offered bands; unassigned shipments are none. |
| `severity_route` | Match the stated severity label; unreadable and unoffered labels are none. |
| `capacity_max` | Compare capacities supplied in option descriptions; select the largest only if it is at least 500 units. |
| `release_line` | Classify the stated release version; unknown and 1.x releases are none. |

This panel is the held-out gate for the post-hoc calibration experiment
(`openkind-bench calibrate-choice`). It is disjoint from
[`joint_calibration_diagnostic.jsonl`](joint_calibration_diagnostic.jsonl) and
from the 2026-09-29 joint-choice panels by task, record, and source group.
The panels were authored before any scoring run of this experiment.

This is a small, templated mechanism diagnostic, not a representative
natural-language benchmark or a promotion gate. Locked calibration parameters
that improve only this panel do not qualify a production change.
