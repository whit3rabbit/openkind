# OpenKind readout, rules and request history study

Status: EXPLORATORY_COMPLETE

Run key: ff6fd4996c079f53dee448296fb55a529d18418d8d849413a2e1ed0e42f6308a

Primary panels complete: True

No model or cache path is promoted. Authored cases share templates and were already inspected.

Raw NLL/Brier are uncalibrated. Offline derived rows have no new latency measurement.

The five-field arm is timed including host derivation; deleting the field also changes the catalogue/prompt.

BF16 versus Q4 changes stored weight precision and backend kernels, not necessarily accumulator precision.

History effects identify request-order dependence; they do not alone identify the underlying kernel/state cause.

Cache speedups are reported only when EVERY matched pair in the complete 96-case shape passes parity and telemetry.

Warm-request timings exclude priming; prime and prime+warm timings are separate.

Repeated calls are averaged within case for paired bootstrap intervals. Repeated fields/calls are not independent samples.

Sources: https://www.privatemode.ai/blog/system-one-from-glm-flash ; https://github.com/thecodacus/llama.cpp/tree/ad129b08d9f134cd298d1f8a85efc52b1b66e18e/tools/parallel-decision

CSV tables and complete responses in jobs/ are the authoritative outputs.
