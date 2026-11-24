Can One Attention Layer Help a Recurrent Model Learn Rules from Examples?

Authors: Eric Park, Max Dunmire, David Veksler, Matthew Okner

Abstract

We investigate whether adding a single causal attention layer to a recurrent sequence model improves its ability to learn new rules from examples presented in a prompt.

We train two small models from scratch: a pure Gated DeltaNet and a hybrid architecture containing Gated DeltaNet blocks and exactly one conventional causal attention layer.

Both models solve synthetic program-execution tasks in which the meaning of a previously unseen command changes between prompts. The models must infer the command's meaning from demonstrations before executing a new program.

We compare paired accuracy across different demonstration counts, investigate the effect of attention-layer placement, and conduct a control experiment in which command meanings are explicitly stated.

Our primary hypothesis is that a single attention layer improves in-context rule learning, particularly when demonstrations are limited. We further investigate whether the advantage depends on attention-layer depth and whether it diminishes when rule identification is unnecessary.

1. Introduction and Background

In-context learning (ICL) refers to a model's ability to learn from examples contained in an input prompt without updating its parameters.

Previous work suggests that both attention-based and recurrent architectures can support in-context learning, although their mechanisms and limitations may differ.

Relevant prior research includes:

Garg et al. (2022), who investigate what Transformers can learn in context using simple function classes.
Olsson et al. (2022), who connect induction heads to the emergence of in-context learning.
Akyürek et al. (2022), who investigate learning algorithms implemented by in-context learners.
Chan et al. (2022), who study how training-data distributions influence emergent in-context learning.
Arora et al. (2024), who investigate recall limitations in efficient language models.
Yang et al. (2025), who introduce Gated Delta Networks and their gated delta-rule memory updates.
Wang et al. (2025), who connect sequence-model architectures with test-time regression.
Daniels et al. (2025), who investigate mechanisms involved in in-context recall.

These findings motivate an investigation into whether conventional attention provides a useful complementary mechanism to recurrent memory.

Rather than comparing large pretrained language models, this project isolates the architectural difference by training small models from scratch under controlled conditions.

The central question is:

Can exactly one attention layer improve a recurrent model's ability to infer new command meanings from demonstrations?

2. Research Questions

We investigate three questions.

RQ1: Attention and Rule Learning

Does adding one causal attention layer improve accuracy when a model must infer a command's meaning from demonstrations?

RQ2: Attention Placement

Does the position of the attention layer within a recurrent architecture affect in-context learning performance?

RQ3: Rule Learning Versus Program Execution

Does the hybrid model's advantage diminish when the command meaning is explicitly provided rather than inferred from examples?

3. Hypotheses
H1: Attention Improves In-Context Rule Learning

The hybrid Gated DeltaNet model will achieve higher paired accuracy than the pure Gated DeltaNet model.

We expect this advantage to be especially pronounced when the number of demonstrations is small.

H2: Attention Placement Matters

Hybrid models with different attention-layer placements will exhibit measurable differences in ICL performance.

The direction and magnitude of this effect will be determined empirically.

H3: Attention's Advantage Is Larger When Rules Must Be Inferred

The hybrid model's advantage over the pure recurrent model will be greater in the demonstration-based ICL task than in the direct-definition control task.

4. Methods
4.1 Task Design

We construct a synthetic program-execution task.

Each prompt contains:

Demonstrations that establish the meaning of a made-up command.
A new program containing that command.
A question asking for the resulting value of a variable.

The command's meaning changes between prompts.

For example, consider the made-up command dax.

In one prompt:

text
a = 2
dax a 3

What is a?


If dax means set, the correct answer is:

text
3


If dax means add, the correct answer is:

text
5


The model cannot rely on a fixed definition of dax. It must infer the intended operation from demonstrations in the current prompt.

4.2 Supported Operations

The initial task should support at least four operations:

set: Replace a variable's value.
add: Add a value to a variable.
subtract: Subtract a value from a variable.
multiply: Multiply a variable by a value.

The generator should support multiple made-up command tokens, with meanings assigned independently for each prompt.

4.3 Demonstration Generation

Generate demonstrations that consistently establish a command's meaning.

Demonstrations must be unambiguous and executable by the reference interpreter.

The number of demonstrations should be configurable.

The experiment should include several demonstration counts, including a zero-demonstration condition where appropriate.

Values, variables, and statement sequences should be randomized.

Avoid making command identity, token frequency, or demonstration formatting predictive of the correct operation.

The same underlying query program should be usable across different command-meaning conditions.

4.4 Program Interpreter

Implement an independent deterministic interpreter.

The interpreter is responsible for:

Parsing program statements.
Initializing variables.
Applying operations.
Executing statements in order.
Returning the queried variable's final value.

The interpreter generates ground-truth answers.

It must be independent of the neural model implementation.

4.5 Dataset Splits

Create independent training, validation, and test datasets.

Use controlled random seeds.

Ensure that test evaluation measures generalization rather than memorization of exact programs.

Where feasible, use distinct underlying program templates or systematically held-out combinations for test evaluation.

Keep the evaluation examples fixed across model comparisons.

5. Model Architectures
5.1 Pure Gated DeltaNet

The baseline model consists of recurrent Gated DeltaNet blocks.

It contains no conventional causal self-attention layers.

Use a consistent embedding layer, tokenizer, output head, and training interface.

5.2 Hybrid Gated DeltaNet

The hybrid model uses the same general architecture but replaces one selected recurrent block with exactly one conventional causal attention layer, or otherwise inserts exactly one attention layer while preserving a matched depth and clearly documenting the change.

The implementation must define the architectural convention precisely and apply it consistently.

5.3 Attention Placement

Implement at least three hybrid configurations:

Early attention: approximately 25% of model depth.
Middle attention: approximately 50% of model depth.
Late attention: approximately 75% of model depth.

Keep other architectural choices fixed.

Record the parameter count of every configuration.

5.4 Parameter Matching

Match the pure and hybrid models as closely as practical in:

Total trainable parameters.
Hidden dimension.
Model depth.
Embedding dimensions.
Output head.
Training budget.

Any unavoidable mismatch must be reported.

The comparison should not accidentally attribute a substantial parameter-count difference to the presence of attention.

6. Training

Train all models from scratch.

Use PyTorch and the Flash Linear Attention library where compatible.

Use the same:

Training-data distribution.
Tokenization.
Optimizer family.
Learning-rate schedule.
Batch-size policy.
Number of training steps.
Validation protocol.

Use multiple independent training seeds.

Target at least five seeds per architecture when computational resources permit.

Save checkpoints and configuration metadata.

Record:

Training loss.
Validation loss.
Learning rate.
Training duration.
Parameter count.
Seed.
Hardware and software environment.
7. Main Experiment: Paired In-Context Learning Accuracy

This is the primary experiment.

For each evaluation case, construct two prompts.

Both prompts must contain:

The same query program.
The same variable initialization.
The same query target.
The same underlying program structure.

However, the demonstrations must establish different meanings for the made-up command.

For example:

Prompt A

Demonstrations establish:

text
dax = set


Query:

text
a = 2
dax a 3
What is a?


Correct answer: 3.

Prompt B

Demonstrations establish:

text
dax = add


Query:

text
a = 2
dax a 3
What is a?


Correct answer: 5.

A test pair is counted as correct only if the model correctly answers both prompts.

7.1 Primary Metric

Define paired accuracy as:

Paired Accuracy=Number of pairs with both answers correctTotal number of evaluated pairs

This metric measures whether the model successfully adapts its interpretation to the demonstrations.

7.2 Additional Metrics

Also report:

Individual answer accuracy.
Accuracy by operation.
Accuracy by demonstration count.
Accuracy by command token.
Accuracy by random seed.
7.3 Demonstration Counts

Evaluate performance across multiple demonstration counts.

The exact counts should be selected based on context length and computational feasibility.

Use the same demonstration counts for all model variants.

Keep query-program length and complexity fixed when comparing demonstration counts.

8. Attention Placement Experiment

Compare the early, middle, and late hybrid configurations.

Keep all other settings fixed.

Evaluate paired accuracy across demonstration counts.

The purpose is to determine whether attention is more useful when applied early, in the middle, or late in the recurrent processing sequence.

Report:

Accuracy by placement.
Parameter counts.
Training and validation losses.
Confidence intervals.
9. Control Experiment: Direct Command Definition

The control experiment tests whether improvements are specific to learning command meanings from examples.

Instead of providing demonstrations, explicitly state the command definition.

For example:

text
dax means add.

a = 2
dax a 3

What is a?


The model must execute the program using the supplied definition.

Use equivalent query programs and operation distributions across the main experiment and control.

Compare the hybrid's advantage in both conditions.

A smaller hybrid advantage in this control would be consistent with the hypothesis that attention particularly helps identify rules from examples.

However, differences between the conditions must be interpreted carefully because prompt structure and token distributions also differ.

10. Statistical Analysis

Use multiple independent training seeds.

Report:

Mean accuracy.
Standard deviation.
95% confidence intervals.
Absolute differences between architectures.
Paired statistical comparisons where appropriate.

The primary comparison is hybrid versus pure Gated DeltaNet paired accuracy.

Use matched evaluation cases.

Do not treat individual prompts as independent model-training runs.

Clearly identify the statistical unit used for each confidence interval and significance test.

Report uncertainty rather than relying exclusively on point estimates.

Do not claim statistical significance without an appropriate test.

11. Visualizations

Produce the following figures:

Paired accuracy versus number of demonstrations.
Individual accuracy versus number of demonstrations.
Pure Gated DeltaNet versus hybrid comparisons.
Early versus middle versus late attention placement.
Training and validation loss curves.
Main ICL experiment versus direct-definition control.

Figures should include:

Axis labels.
Legends.
Consistent colors.
Appropriate confidence intervals.
Descriptive titles.

Export each figure as PNG and PDF.

12. Expected Outcomes

We expect the hybrid architecture to outperform pure Gated DeltaNet on demonstration-based rule learning, particularly with few examples.

We also expect the placement of attention to influence performance.

Finally, we expect the hybrid's advantage to be smaller when the command definition is stated explicitly.

These are hypotheses, not predetermined results.

All outcomes, including negative or inconclusive findings, should be reported.

13. Limitations

The synthetic task provides experimental control but does not fully represent natural-language in-context learning.

The results may depend on:

Task construction.
Tokenization.
Model size.
Training distribution.
Demonstration formatting.
Attention implementation.
Recurrent architecture configuration.

The study should avoid claiming that its findings automatically generalize to large language models or all forms of ICL.

14. References
Garg et al. (2022). What Can Transformers Learn In-Context? A Case Study of Simple Function Classes. https://arxiv.org/abs/2208.01066
Olsson et al. (2022). In-context Learning and Induction Heads. https://arxiv.org/abs/2209.11895
Akyürek et al. (2022). What learning algorithm is in-context learning? Investigations with linear models. https://arxiv.org/abs/2211.15661
Chan et al. (2022). Data Distributional Properties Drive Emergent In-Context Learning in Transformers. https://arxiv.org/abs/2205.05055
Arora et al. (2024). Zoology: Measuring and Improving Recall in Efficient Language Models. https://arxiv.org/abs/2312.04927
Yang et al. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. https://arxiv.org/abs/2412.06464
Wang et al. (2025). Test-time regression: a unifying framework for designing sequence models with associative memory. https://arxiv.org/abs/2501.12352
Daniels et al. (2025). Decomposing Prediction Mechanisms for In-Context Recall. https://arxiv.org/abs/2507.01414
15. Definition of Research Success

The project succeeds if it produces a reproducible and statistically defensible comparison between pure Gated DeltaNet and hybrid Gated DeltaNet architectures.

Success does not require the hypotheses to be confirmed.

The most important outcome is a reliable answer to whether one conventional attention layer provides a measurable advantage in learning new command meanings from examples.
