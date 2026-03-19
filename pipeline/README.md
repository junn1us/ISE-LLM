# SmartFixture-LLM4Teach Training Session Guide

## Scope
This document defines a reproducible procedure for distillation-then-PPO training, multi-seed comparison, and leaderboard generation with 95% confidence intervals.

## Environment
Use the project virtual environment:

- Python executable: `/Users/junn1us/Library/Mobile Documents/com~apple~CloudDocs/ISYE URP/.venv-1/bin/python`

## Single-run training command
A full run performs teacher distribution collection, student distillation, and PPO fine-tuning.

```bash
"/Users/junn1us/Library/Mobile Documents/com~apple~CloudDocs/ISYE URP/.venv-1/bin/python" ppo_finetune.py \
  --pretrain-teacher \
  --force-scratch \
  --iters 100 \
  --episodes-per-iter 20 \
  --seed 16 \
  --out ppo_fixture_policy_seed16_it100.pt
```

## Multi-seed training command
Execute the reproducible multi-seed runner:

```bash
bash scripts/run_multiseed_100iter.sh
```

Default seeds in the script: `16, 17, 18`.

## Leaderboard evaluation with confidence intervals
Evaluate checkpoints with stochastic policy rollout and compute 95% confidence intervals.

```bash
"/Users/junn1us/Library/Mobile Documents/com~apple~CloudDocs/ISYE URP/.venv-1/bin/python" scripts/evaluate_leaderboard.py \
  --checkpoints \
    ppo_fixture_policy_seed16_it100.pt \
    ppo_fixture_policy_seed17_it100.pt \
    ppo_fixture_policy_seed18_it100.pt \
    ppo_fixture_policy_seed16.pt \
  --episodes 300 \
  --eval-seed 16 \
  --out-csv leaderboard_100iter_seeds.csv
```

## Recommended artifacts to commit
Code and documentation:
- `smartfixture_llm4teach/core/action_space.py`
- `smartfixture_llm4teach/core/fixture_core.py`
- `smartfixture_llm4teach/envs/student_env.py`
- `smartfixture_llm4teach/envs/teacher_env.py`
- `smartfixture_llm4teach/teacher/translator.py`
- `smartfixture_llm4teach/teacher/parser.py`
- `smartfixture_llm4teach/teacher/llm_adapter.py`
- `ppo_finetune.py`
- `collect_dataset.py`
- `scripts/run_multiseed_100iter.sh`
- `scripts/evaluate_leaderboard.py`
- `README_TRAINING_SESSION.md`

Evaluation outputs:
- `leaderboard_100iter_seeds.csv`
- `leaderboard_100iter_seeds.txt`

Model checkpoints are typically large and may be tracked via releases, artifact storage, or LFS according to repository policy.
