# Deployment & Compilation Architecture

This repository uses `stickytape` to compile our modular source code into a single, flattened bundle script optimized for serverless remote execution targets (e.g., Kaggle GPU kernels). Because `stickytape` relies on static code analysis to resolve the dependency graph, all contributors must abide by the following structural invariants:

## 1. Absolute Import Enforcement

- Do **not** use relative dot-imports (e.g., `from .tracking import Tracker`).
- All internal module lookups must use absolute package-root targeting (e.g., `from roofseg.common.tracking import Tracker`).

For a detailed reference on how the static dependency parsing behaves, please review the official documentation: 👉 [Stickytape Compiler Documentation & Reference](https://github.com/mwilliamson/stickytape)
