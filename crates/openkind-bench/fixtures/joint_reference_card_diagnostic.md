# Reference-card intervention

[`joint_reference_card_diagnostic.jsonl`](joint_reference_card_diagnostic.jsonl)
contains 24 labeled Choice rows in six source groups. Within each group,
state, instructions, and descriptions of `alpha`, `beta`, `gamma`, and
`__none__` are identical. Only a fourth real option, `reference`, changes:
its card maps the state's code to alpha, beta, gamma, or the absent action
delta. Gold is respectively alpha, beta, gamma, or `__none__`. Each gold
appears six times. The reference option explicitly says it must never win.

Independent fixed-temperature logits cannot change relative preferences
among the unchanged action options in response to the card. The fitted none
head can still change none mass because it reads summaries of all candidate
logits. Joint prompts can inspect the card before scoring the actions.

This separate panel was authored after the first 96-case diagnostic run
started, without fitting or changing either readout. It is a mechanism
intervention motivated by the article, not an independent natural-task
quality gate. Source-group bootstrap intervals cover these six authored
groups only. Historical gate and final partitions remain excluded.
