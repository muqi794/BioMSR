# Ablation Experiments

This directory contains structural ablations only. The full model should be run separately with `train_model.py`.

Variants:
- `w_o_sim`: remove the similarity branch and keep only heterogeneous representations.
- `w_o_horv_plus_sim`: remove the high-order relational view and the similarity branch, leaving only the low-order relational view.
- `w_o_lorv_plus_sim`: remove the low-order relational view and the similarity branch, leaving only the high-order relational view.
- `w_o_horv`: remove the high-order relational view while keeping similarity and low-order views.
- `w_o_lorv`: remove the low-order relational view while keeping similarity and high-order views.

Run a single ablation from the project root, for example:

```bash
python ablations/w_o_horv/train.py --epochs 100
```

Run the full ablation suite from a single entry point:

```bash
python ablations/run.py --epochs 100
```

Run only selected variants:

```bash
python ablations/run.py --variants w_o_sim w_o_horv w_o_lorv --epochs 100
```

Run the complete model separately:

```bash
python train_model.py --epochs 100
```
