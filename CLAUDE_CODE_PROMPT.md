Autonomous Research Implementation: Claude Code Instructions
Project

Can One Attention Layer Help a Recurrent Model Learn Rules from Examples?

You are an autonomous machine learning research engineer. Your responsibility is to independently implement, execute, evaluate, and document the research project described in RESEARCH_PROPOSAL.md.

Your objective is not merely to produce code. Your objective is to conduct a reproducible empirical investigation that answers the research questions with statistically meaningful experimental evidence.

You should work independently, make reasonable engineering decisions, identify implementation problems, debug them, and continue until the project reaches its defined completion criteria.

Do not stop after writing an implementation plan.

1. Primary Objective

Investigate whether adding exactly one causal attention layer to a recurrent Gated DeltaNet architecture improves in-context learning (ICL) when the model must infer a previously unseen command definition from demonstrations.

Compare:

Pure Gated DeltaNet: A recurrent model without conventional attention layers.
Hybrid Gated DeltaNet: The same recurrent architecture with exactly one causal attention layer.
Hybrid placement variants: The same hybrid architecture with the attention layer inserted at different depths.

Evaluate whether attention provides its greatest benefit when the model must infer command semantics from examples, rather than when those semantics are explicitly provided.

The central experimental metric is paired accuracy.

2. Autonomous Execution Requirements

You should conduct the project from beginning to end.

You are authorized to:

Inspect the existing repository and environment.
Create and modify project files.
Install appropriate dependencies when necessary.
Implement model architectures.
Generate synthetic datasets.
Train models.
Run evaluation experiments.
Run multiple random seeds.
Generate plots and tables.
Analyze results.
Identify implementation errors and correct them.
Write a final research report.

Do not ask for permission for routine implementation decisions.

If a dependency is unavailable, investigate compatible alternatives before abandoning the experiment.

If GPU resources are unavailable, implement an appropriate CPU-compatible configuration or reduce model size while preserving the experimental design.

Do not silently change the research question or eliminate important experimental conditions to make implementation easier.

When computational limitations require a reduced experiment, document the deviation and its consequences.

3. Initial Repository Inspection

Before implementing anything:

Inspect the repository structure.
Identify existing code, datasets, configurations, and dependencies.
Check Python and PyTorch versions.
Check CUDA availability and GPU memory.
Determine whether Flash Linear Attention is installed and compatible.
Identify available computational resources.
Create a project implementation plan.

Create or update:

IMPLEMENTATION_PLAN.md
EXPERIMENT_LOG.md
requirements.txt or an appropriate environment configuration.

Do not overwrite useful existing work without first inspecting it.

4. Implementation

Implement the complete experiment described in RESEARCH_PROPOSAL.md.

A. Synthetic Task Generator

Build a deterministic, configurable program generator.

Each generated example must contain:

A command definition established through demonstrations.
A sequence of demonstrations.
A query program.
A ground-truth answer.

Use made-up commands such as dax, wug, and blick.

Their meanings should vary across prompts.

At minimum, support:

set
add
subtract
multiply

Ensure that the same command can represent different operations in different prompts.

Randomize:

Initial variable values.
Variable assignments.
Statement ordering.
Query targets.
Demonstration examples.

Use an independent interpreter to calculate correct answers.

Prevent accidental data leakage between training, validation, and test sets.

Make dataset generation reproducible through explicit random seeds.

B. Model Architectures

Implement two approximately parameter-matched model families.

Model A: Pure Gated DeltaNet

Uses recurrent Gated DeltaNet blocks.
Contains no conventional self-attention layers.
Uses the same tokenizer, embeddings, output head, and general training interface as the hybrid.

Model B: Hybrid Gated DeltaNet

Uses Gated DeltaNet blocks.
Contains exactly one causal self-attention layer.
Supports configurable attention placement.

Implement at least three attention placements:

Early: approximately one-quarter of the model depth.
Middle: approximately one-half of the model depth.
Late: approximately three-quarters of the model depth.

Keep the remaining architecture as consistent as possible.

Document parameter counts, architectural differences, and any unavoidable deviations.

Use Flash Linear Attention's Gated DeltaNet implementation if compatible. Otherwise, investigate an equivalent implementation and clearly document the substitution.

Do not label an architecture Gated DeltaNet unless its implementation actually follows the relevant Gated DeltaNet formulation.

C. Training

Train all models from scratch.

Use:

Identical training data distributions.
Identical tokenization.
Identical optimizer configurations where feasible.
Identical training-step budgets.
Comparable parameter counts.
Multiple independent random seeds.

Create centralized configuration files rather than hardcoding hyperparameters.

Track:

Training loss.
Validation loss.
Learning rate.
Training steps.
Model parameter count.
Training duration.
Random seed.
Hardware configuration.

Save model checkpoints and training metadata.

Implement checkpoint recovery and training resumption.

D. Evaluation

The main evaluation must vary the number of demonstrations while keeping query-program complexity fixed.

For each test case:

Construct a query program.
Construct two prompts with identical query programs.
Give the prompts different demonstrations.
Ensure the demonstrations establish different meanings for the same made-up command.
Evaluate both prompts independently.
Count the test case as correct only if both answers are correct.

Define:

paired_accuracy = number of correctly answered pairs / total number of pairs

Evaluate multiple demonstration counts.

Use identical test cases across architectures and seeds wherever possible.

Also evaluate:

Ordinary individual-answer accuracy.
Paired accuracy.
Accuracy by command meaning.
Accuracy by demonstration count.
Accuracy by attention placement.
Training and validation loss.
E. Control Experiment

Implement a control condition where the command meaning is explicitly stated in the prompt.

For example:

dax means add.

The model must then execute the query program without having to infer the command definition from demonstrations.

Compare the hybrid's advantage in the primary ICL task against its advantage in this direct-definition control.

This experiment is necessary to distinguish improved rule acquisition from generally improved program execution.

5. Experimental Design and Statistical Analysis

Use multiple training seeds. Aim for at least five independent seeds if computational resources permit.

Use paired test examples across model comparisons.

Report:

Mean accuracy.
Standard deviation across seeds.
95% confidence intervals.
Absolute accuracy differences.
Relative improvements where appropriate.

For comparisons, use paired statistical analysis when the evaluation design permits it.

Do not treat individual prompts as independent training seeds.

Clearly distinguish:

Variation across random seeds.
Variation across evaluation examples.
Variation across training runs.

Avoid making claims of statistical significance without an appropriate statistical test.

If the sample size is insufficient, state that explicitly.

6. Required Visualizations

Generate publication-quality plots.

At minimum:

Paired accuracy versus demonstration count.
Individual accuracy versus demonstration count.
Comparison of pure Gated DeltaNet against hybrid variants.
Attention placement comparison.
Training and validation loss curves.
ICL task versus direct-definition control comparison.

Include confidence intervals where appropriate.

Use consistent colors, labels, scales, and legends.

Save figures in PNG and PDF formats.

7. Required Project Structure

Organize the repository in a maintainable way. A suggested structure is:

text
project/
├── CLAUDE_CODE_PROMPT.md
├── RESEARCH_PROPOSAL.md
├── IMPLEMENTATION_PLAN.md
├── EXPERIMENT_LOG.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── pure_deltanet.yaml
│   └── hybrid.yaml
├── src/
│   ├── data/
│   │   ├── generator.py
│   │   ├── interpreter.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── pure_deltanet.py
│   │   ├── hybrid.py
│   │   └── common.py
│   ├── training/
│   │   ├── train.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── paired_accuracy.py
│   │   └── statistics.py
│   └── utils/
│       ├── reproducibility.py
│       └── logging.py
├── scripts/
│   ├── generate_data.py
│   ├── run_experiments.py
│   ├── evaluate_models.py
│   └── generate_figures.py
├── tests/
│   ├── test_generator.py
│   ├── test_interpreter.py
│   ├── test_models.py
│   └── test_evaluation.py
├── data/
├── checkpoints/
├── results/
│   ├── tables/
│   ├── figures/
│   └── raw/
└── reports/
    └── FINAL_REPORT.md


You may modify this structure if a better organization is justified.

8. Testing and Verification

Write tests before relying on experimental results.

Verify:

The interpreter correctly executes every supported operation.
The generator produces valid programs.
Different prompts can assign different meanings to the same command.
Paired prompts have identical query programs.
Paired prompts have different demonstrated command meanings.
Training and test data do not leak.
Models produce valid output formats.
Attention masks prevent access to future tokens.
The pure model contains no conventional attention layer.
Each hybrid contains exactly one conventional attention layer.
Attention placement matches the configuration.
Paired accuracy is calculated correctly.
Evaluation is deterministic under fixed seeds.

Run the complete test suite before starting large experiments.

9. Research Integrity

This project is an empirical investigation, not an exercise in confirming the hypothesis.

You must:

Report negative results.
Report inconclusive results.
Preserve raw experimental outputs.
Never fabricate metrics.
Never invent completed training runs.
Never claim an experiment was executed when it was only implemented.
Record deviations from the original proposal.
Distinguish observed results from interpretations.
Explain limitations of the synthetic task.
Investigate unexpected results rather than suppressing them.

If the hybrid performs worse than the pure recurrent model, that is a valid research finding.

Do not change evaluation criteria after seeing the results.

10. Final Deliverables

The project is complete when the following deliverables exist:

Implementation
Working synthetic task generator.
Independent program interpreter.
Pure Gated DeltaNet model.
Hybrid model with one attention layer.
Configurable attention placement.
Reproducible training pipeline.
Evaluation pipeline.
Automated tests.
Experimental Results
Trained checkpoints.
Raw evaluation outputs.
Training logs.
Statistical analysis.
Accuracy tables.
Publication-quality figures.
Documentation
README.md with setup and execution instructions.
IMPLEMENTATION_PLAN.md.
EXPERIMENT_LOG.md.
reports/FINAL_REPORT.md.
Final Research Report

The report must include:

Abstract.
Introduction.
Research questions.
Hypotheses.
Methods.
Model architectures.
Dataset generation.
Training configuration.
Experimental design.
Main results.
Attention placement results.
Control experiment.
Statistical analysis.
Discussion.
Limitations.
Conclusion.
Reproducibility instructions.

Include actual tables and figures.

Explicitly state which experiments were completed and which were not.

11. Execution Strategy

Proceed in the following order:

Phase 1 — Environment and planning

Inspect the environment, dependencies, and available hardware. Create the implementation plan.

Phase 2 — Task and data

Implement the program interpreter, synthetic data generator, and tokenizer. Test them thoroughly.

Phase 3 — Model implementation

Implement and verify both architectures, including attention placement variants.

Phase 4 — Pipeline validation

Run small-scale training and evaluation experiments to identify bugs.

Phase 5 — Main experiments

Train all models using the agreed experimental configurations and random seeds.

Phase 6 — Evaluation

Run paired evaluation, direct-definition controls, and statistical analysis.

Phase 7 — Visualization

Generate tables and publication-quality figures.

Phase 8 — Research report

Interpret the actual findings and produce the final report.

Phase 9 — Final verification

Verify that all documented commands work and all reported metrics can be traced to raw experimental results.

Update EXPERIMENT_LOG.md throughout the process.

12. Completion Criteria

Do not declare completion merely because the code runs.

Completion requires:

Correct implementations.
Passing tests.
Reproducible experimental configurations.
Completed experiments or clearly documented resource-related limitations.
Actual evaluation results.
Statistical analysis.
Figures and tables.
A final research report.

At the end, provide a concise summary of:

What was implemented.
What experiments were executed.
What results were obtained.
Whether the hypothesis was supported.
What remains incomplete.
Exact commands needed to reproduce the experiments.

Begin by inspecting the repository and environment. Then proceed autonomously through the research workflow.
