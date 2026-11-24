# Can one attention layer help a model remember?

Imagine a list with `dax = red` and `wug = blue`.
Then ask, “What goes with `dax`?”
The answer is `red`.

A Gated DeltaNet (GDN) keeps a memory that stays the same size as the list grows.
An attention layer can look back at the list.
We wanted to see if adding one attention layer helps when the list gets long.

## What changed and why

1. **We took out the math.**
   The old task made the models remember a number and use it in a math problem.
   The model needed both skills to get the answer.
   The GDN and the GDN with one attention layer each got about 7% right, even when we gave them the rule.
   A guess had a 7.7% chance of being right.
   Those scores could not tell us how much the models remembered.
   [Math run results](reports/FINAL_REPORT.md#161-why-the-mod-13-pilot-failed)
2. **We switched to a lookup task called MQAR.**
   The model reads 4 to 256 pairs and answers questions like the one above.
   The answer is the value next to the word in the list.
   There are 64 possible values, so a guess is right about 1.6% of the time.
   [Lookup task](reports/FINAL_REPORT.md#162-multi-query-associative-recall)
3. **We added a check before the comparison.**
   For this check, we trained a transformer, which uses attention in every layer.
   It had to get at least 90% right on lists of 16 pairs before we compared models.
   We also added a setting for a smaller GDN memory to test what happens when space runs out.
   [Study plan](IMPLEMENTATION_PLAN.md)
4. **We made the models smaller.**
   We estimated that the full plan would take 15 to 20 hours on one GPU.
   We cut the model from eight layers to four.
   We made each layer half as wide (128 instead of 256).
   Once training got going, each step took 0.058 seconds, down from 0.196 seconds.
   [Timing logs](EXPERIMENT_LOG.md)

## What happened

The small transformer got **11.25%** right on lists of 16 pairs after 1,500 steps.
It missed the 90% mark, so we stopped before the planned comparison.
We scored the small runs on questions we held back from training.
[Transformer scores](results/mqar_compact/tables/validation_accuracy.csv)

In a short run to measure speed, a small model with one attention layer got **97.75%** right on lists of 16 pairs.
That shows the lookup task can be learned.
The models started with different random settings and trained for different numbers of steps.
We still need a fair comparison to tell if one attention layer helps.
[Full results](reports/FINAL_REPORT.md#163-compact-validation-study)

## Run it again

- [Make the MQAR chart from the saved logs](reports/FINAL_REPORT.md#163-compact-validation-study).
- [Run the original experiments](reports/FINAL_REPORT.md#15-reproducibility).
