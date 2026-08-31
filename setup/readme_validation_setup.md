# Validation and Setup

These files define the software environment, verify the 12-anchor implementation, and document the reproducible training workflow.

## Setup Flow

```text
install dependencies
    -> run quick integration checks
    -> validate 12-anchor interfaces and artifacts
    -> follow the training guide
```

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Lists the Python dependencies for numerical computing, PyTorch training, Gymnasium environments, plotting, and Hugging Face model integration. |
| `test_implementation.py` | Runs fast integration checks for imports, the 12-anchor environment, teacher context generation, and probability parsing. Its `--full` option also runs a short PPO workflow. |
| `validate_12_anchor_setup.py` | Performs detailed checks of anchor positions, observation and action dimensions, teacher interfaces, dataset shapes, probability normalization, and student-checkpoint compatibility. |
| `README.md` | Provides the main project overview, installation instructions, pipeline commands, configuration details, results, and troubleshooting guidance. |
| `README_TRAINING_SESSION.md` | Records the reproducible single-run and multi-seed training commands, leaderboard evaluation procedure, default seeds, and recommended public artifacts. |

## Typical Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python test_implementation.py --quick
python validate_12_anchor_setup.py
```

To include the short training check:

```bash
python test_implementation.py --full
```

## Validation Coverage

- Twelve expected anchor coordinates: eight boundary and four interior.
- Observation size of 15 features and action size of 13 including NOOP.
- Teacher context, logits, and normalized action probabilities.
- Optional dataset compatibility for `teacher_data_12_X.npy` and `teacher_data_12_P.npy`.
- Optional 12-action student checkpoint compatibility for `student_policy_init.pt`.

## Reproducibility Notes

- Run validation commands from the repository root.
- Missing datasets and checkpoints are skipped by the validation suite; generate them before validation when artifact checks are required.
- `requirements.txt` currently does not pin package versions; a public release should include tested versions or a lock file.
- `README_TRAINING_SESSION.md` contains a machine-specific Python path that should be replaced with portable environment commands before publication.
