# Joint Choice diagnostic panel

[`joint_choice_diagnostic.jsonl`](joint_choice_diagnostic.jsonl) contains 96
authored Choice questions, all with explicit `__none__` descriptions. Each of
four tasks has six examples for each gold label (`alpha`, `beta`, `gamma`,
`__none__`). Cases sharing a task and named record share `source_group`, giving
24 groups for paired bootstrap resampling.

| Task | Evidence and gold rule |
|---|---|
| `state_fact` | Match the parcel's stated status; cancelled is absent from the options. |
| `negation` | Match the stated valve color while ignoring explicitly negated colors; yellow is absent. |
| `threshold` | Classify a numeric risk score into the offered inclusive ranges; -1 is outside every range. |
| `option_reference` | Compare measurements supplied in option descriptions; select the largest only if it is at least 80. |

The last task directly tests the article's observation that options can supply
evidence needed to compare rivals. The independent renderer sees only one
option's measurement per forward. The joint renderer sees every measurement.
The comparison changes the prompt and readout together, so it cannot isolate
which change causes an improvement.

This is a small, templated mechanism diagnostic authored before scoring.
It is not a representative natural-language benchmark or a promotion gate.
Historical calibration gates and final partitions are not included. Do not
tune a temperature or threshold on this panel and report it as held-out quality.
