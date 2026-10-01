# Evaluation assets

This directory separates reproducible evaluation inputs and reports from
runtime artifacts. Generated JSON is intentionally not overwritten by the
CLI; each report records its source run directories and configuration hashes.

```text
manifests/       task, run, and environment manifests
results/         raw, normalized, and paired results
failure_cases/   trace-linked recovery and tool failure cases
reports/         human-readable and machine-readable summaries
```

Each result set includes its dataset identifier, task manifest, model and
Harness configuration, resource budget, and evaluator version. Report task
resolution, recovery events, and execution cost separately. Keep deterministic
smoke outputs in their own run directories.
