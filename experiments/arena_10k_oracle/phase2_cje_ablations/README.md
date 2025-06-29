# Phase 2: CJE Ablations - Fresh Start

This directory contains the clean implementation for running CJE ablations on Arena 10K data.

## Current Status

Starting fresh after fixing critical teacher forcing bug that affected 708 samples (7.08%).

## Key Components

- `configs/ablations/` - Configuration files for different estimators

## Next Steps

1. Recompute all log probabilities using `RobustTeacherForcing`
2. Run fresh CJE analysis with clean data
3. Validate that model performance rankings are correct

The robust teacher forcing implementation is now in the core library at `cje.utils.teacher_forcing`.