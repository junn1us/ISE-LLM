# Reproducible Run Wrappers

These scripts automate multi-seed PPO training, checkpoint evaluation, and aggregation of the 1000-iteration MF5 experiments.

## Data Flow

```text
multi-seed training
    -> PPO checkpoints
    -> leaderboard evaluation
    -> per-run deformation metrics
    -> cross-batch summary
```

## Files

| File | Purpose |
|---|---|
| `scripts/run_multiseed_100iter.sh` | Runs teacher-pretrained PPO for seeds 16, 17, and 18 using 100 iterations and 20 episodes per iteration. It writes one checkpoint per seed. |
| `scripts/evaluate_leaderboard.py` | Evaluates one or more PPO checkpoints with stochastic rollouts, computes mean return and final RMSE with 95% confidence intervals, ranks the models, and saves a CSV leaderboard. |
| `scripts/summarize_mf5_1000iter_batches.py` | Aggregates deformation metric and pairwise-comparison CSV files for MF5 batches 5–9. It reports per-batch statistics, global statistics, and the best and worst RMSE runs. |

## Typical Usage

```bash
bash scripts/run_multiseed_100iter.sh

python scripts/evaluate_leaderboard.py \
  --checkpoints ppo_fixture_policy_seed16_it100.pt \
                ppo_fixture_policy_seed17_it100.pt \
                ppo_fixture_policy_seed18_it100.pt \
  --episodes 300 \
  --out-csv leaderboard_100iter_seeds.csv

python scripts/summarize_mf5_1000iter_batches.py
```

## Outputs

- `ppo_fixture_policy_seed*_it100.pt`: trained multi-seed checkpoints.
- `leaderboard_100iter_seeds.csv`: ranked checkpoint metrics and confidence intervals.
- `outputs/deformation_summary_1000iter_mf5_batches5to9.csv`: per-batch aggregate metrics.
- `outputs/deformation_summary_1000iter_mf5_batches5to9.txt`: readable global and per-batch summary.

## Reproducibility Notes

- Run commands from the repository root with dependencies installed.
- `run_multiseed_100iter.sh` currently contains an absolute local project and Python path; replace these before running on another machine.
- Use identical environment settings, checkpoint architectures, seeds, and evaluation episode counts when comparing results.
- The MF5 summarizer expects its input CSV files in `outputs/` with the batch 5–9 naming convention defined in the script.
