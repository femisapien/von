# Fine-tuning Von on your own decisions

Three levels, cheapest first. Stop at the first one that fixes your problem.

| level | what changes | what it fixes | cost |
|---|---|---|---|
| 1. `von calibrate` | temperature map only | over/under-confidence; never a pick | CPU, minutes |
| 2. continue-train | head + encoder weights, from von-1.2 | wrong picks on an English domain Von has not seen | 1 GPU, ~1 h |
| 3. new trunk | fresh head on another MLM encoder | another language, longer context, smaller model | 1 GPU, 1–4 h, needs the base corpus |

Levels 2 and 3 use the same trainer (`training/train_option_marker.py`) and the same row
format. Nothing here needs AWS; the `launch_*.py` scripts in `training/` are our own
spot-instance wrappers and you can ignore them.

## Row format

One JSON object per line. Same shape Von serves, plus the answer:

```json
{"state": "Invalid API key · Please run /login",
 "question": "What kind of failure is this?",
 "options": [{"id": "auth",       "description": "Login or credentials problem"},
             {"id": "code_error", "description": "A bug or failing test in the code"},
             {"id": "rate_limit", "description": "Too many requests"}],
 "label": "auth"}
```

- `options`: 2–32 described options. The description is what the model reads; the id is only
  the answer key. Noul is a 2-option Choice (`{"id":"yes",...},{"id":"no",...}`); Score is an
  ordered list of levels.
- `label` must equal one option id (the loader refuses the file otherwise).
- `target` (optional): a probability per option id for soft labels, e.g. preference strength.
  Brier loss uses it; otherwise the label is one-hot.
- Shuffle option order across rows. Von is order-invariant by construction
  (`--independent_options`), but balanced data keeps the head honest.
- Keep a held-out split that the trainer never sees and score it with
  `benchmarks/gate_standard.py`-style paired tests before believing a number. Trainer
  validation accuracy is not a ship criterion.

Converting `von calibrate` label files is mechanical: `question.criteria` becomes
`options`, `gold` becomes `label`.

## Level 2: continue-train from von-1.2 (English)

A few hundred to a few thousand rows of your domain, mixed with replay from the base
corpus so the model does not forget everything else. This is the recipe behind every
von-1.3/1.4 experiment in `results/`.

```bash
# 1. weights
hf download wfzyx/von --local-dir checkpoints/von-1.2

# 2. replay rows (any slice of the public mix; 10–30k is enough)
uv run python training/prepare_universal_dataset.py --output_dir data_replay --max_train 20000 --val_samples 1000

# 3. your rows + replay, shuffled
cat your_train.jsonl data_replay/train.jsonl | shuf > data_ft/train.jsonl
cp your_val.jsonl data_ft/val.jsonl

# 4. train (A10G 24 GB: batch 4 × accum 4; drop to batch 1 × accum 16 for rows over ~4k chars)
uv run python training/train_option_marker.py \
  --init_checkpoint checkpoints/von-1.2 --base_model_id checkpoints/von-1.2 \
  --train_data data_ft/train.jsonl --val_data data_ft/val.jsonl \
  --output_dir checkpoints/von-ft --epochs 1 --lr 1.5e-5 \
  --batch_size 4 --grad_accum_steps 4 --independent_options --digit_split

# 5. refit confidence on the new head, then serve
uv run von calibrate your_val.jsonl --checkpoint checkpoints/von-ft
VON_CHECKPOINT_DIR=$PWD/checkpoints/von-ft uv run von serve
```

`--init_checkpoint` loads `option_marker.pt` (encoder + head); `--base_model_id` only
supplies the tokenizer/config, so point both at the same directory. `--independent_options`
and `--digit_split` must match what the checkpoint was trained with (von-1.2: both on);
`marker_calibration.json` records the first; nothing checks the second, so a mismatch silently
trains a different tokenisation than the one served. Pass both.

Expectations, from our own runs: a targeted domain with clean labels moves a lot (Jeff's
adapters report +30 pp on-task from ~5k rows; our judge-mix run moved one family +6 items of
12 at p=0.03). Sealed, general benchmarks do not move — see `results/v1.4/`. Fine-tune for
your task, not for the board.

## Level 3: another encoder (German, multilingual, smaller)

Von's encoder is `answerdotai/ModernBERT-large`, English only. Non-English input gets token
matching with confidence it has not earned. A continue-train on German labels improves the
matching, not the understanding; for real German you change the trunk.

The trainer accepts any Hugging Face MLM encoder with a mask token (`AutoModel` +
`tokenizer.mask_token`). Candidates that keep Von's shape (8k context, ModernBERT
architecture, Apache-2.0, so OpenVINO export, `calibrate` and chains carry over unchanged):

| trunk | params | languages | notes |
|---|---|---|---|
| `jhu-clsp/mmBERT-base` | 307M | 1,800+ incl. German | ModernBERT architecture; the obvious German pick |
| `jhu-clsp/mmBERT-small` | 140M | same | on-device budget |
| `EuroBERT/EuroBERT-210m` / `-610m` | 210M / 610M | European languages | different architecture; needs an export check |

A fresh head needs the full base corpus, not just your labels. Build it, add your rows, train
one epoch:

```bash
uv run python training/prepare_universal_dataset.py --output_dir data_universal \
    --max_train 290000 --synthetic_n 40000 --val_samples 5000
cat your_german_train.jsonl >> data_universal/train.jsonl

uv run python training/train_option_marker.py \
  --base_model_id jhu-clsp/mmBERT-base \
  --train_data data_universal/train.jsonl --val_data data_universal/val.jsonl \
  --output_dir checkpoints/von-de --epochs 1 --lr 3e-5 \
  --batch_size 8 --grad_accum_steps 2 --independent_options --digit_split
```

Cost: the 290k-row epoch on ModernBERT-large takes ~1 h on 4×A10G (~$12); mmBERT-base is
roughly 2× cheaper. The universal corpus is English; a German model trained on it plus a
few thousand German rows will be an English decision model that reads German, which is
usually what a mixed-language inbox needs. A properly German corpus is a data project, not
a training one — the generators in `training/generate_*.py` are template-driven and can be
translated.

The result serves exactly like Von (`VON_CHECKPOINT_DIR=... von serve`); the only runtime
difference is the tokenizer that comes with the trunk. Calibration must be refitted
(`von calibrate`) — the shipped map is fitted to ModernBERT-large logits and is wrong for
any other encoder.

## Gates

Whatever level you use, keep the discipline that kept Von honest:

- lock a held-out set before training, never touch it afterwards;
- compare old vs new with a paired test (`benchmarks/stat_gate.py`, McNemar + MDE), not
  with two accuracies;
- do not fit on anything you later report.
