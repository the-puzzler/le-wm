# laleworldmodel

Simultaneous latent-action and world-model learning from passive video for controllable prediction.

This repository is the codebase for **la leWorldModel**: a latent-action extension of LeWorldModel that removes the need for environment actions during world-model training. Instead of conditioning the dynamics model on labelled actions, the model learns a small discrete latent action vocabulary directly from video and trains that jointly with the world model in a single run.

Blog post: [la leWorldModel](https://the-puzzler.github.io/?p=la-leworldmodel)

## What This Repo Is

The original LeWorldModel setup assumes access to real environment actions. This repo changes that assumption.

`laleworldmodel`:

- trains from passive video trajectories
- infers discrete latent actions from neighbouring latent states
- quantizes them with a small codebook
- conditions the world model on those learned latent actions
- trains the latent-action model and world model jointly, end to end

In this codebase, the core latent-action path is implemented through:

- an inverse-dynamics style latent-action predictor
- a vector-quantized codebook
- an action-conditioned predictor in latent space
- a separate visual decoder trained afterwards for rendering imagined rollouts
- an optional action translator for PushT, mapping latent actions back to executable continuous actions

## Method Overview

At a high level:

1. Encode video frames into latent states.
2. Infer a discrete latent action between adjacent latent states.
3. Quantize that action into one of a small number of reusable action primitives.
4. Predict the next latent state conditioned on the current latent history and the learned latent action.
5. Train everything jointly with the world-model objective rather than a separate pixel-space latent-action pretraining stage.

This is the main distinction from staged latent-action pipelines such as Genie-style approaches: here the latent action model lives directly in the same latent space as the world model and is optimized together with it.

## Current Scope

The main implemented workflow in this repo is centered on **PushT**.

The repo currently contains:

- joint training for the latent-action world model
- training for a visual decoder that maps latent states back to pixels
- training for an action translator used for PushT control
- evaluation and planning utilities
- rendering utilities for imagined trajectories

## Repository Layout

- `train.py`: main training loop for the joint latent-action world model
- `config.py`: single-file configuration for training, decoder training, translator training, and evaluation
- `jepa.py`: JEPA/world-model wrapper
- `module.py`: model components, including predictor, inverse dynamics, vector quantizer, decoder, and action translator
- `eval.py`: planning and evaluation entrypoint
- `scripts/train_decoder.py`: train the visual decoder after world-model training
- `scripts/train_action_translator.py`: train the PushT latent-to-real action translator
- `scripts/render_eval_imagined_trajectories.py`: decode and render imagined planning rollouts
- `scripts/download_data.py`: helper for downloading the PushT dataset

## Installation

This repo uses `uv` and depends on `stable-pretraining` and `stable-worldmodel`.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

If you prefer direct installation:

```bash
uv pip install -e .
```

## Data

By default the repo expects datasets under:

```bash
data/stablewm
```

The default training setup in [`config.py`](/Users/matteo/Documents/le-wm/config.py) uses:

- dataset: `pusht_expert_train`
- frameskip: `5`

You can download PushT with:

```bash
uv run python scripts/download_data.py --dataset pusht
```

That writes:

```bash
data/stablewm/pusht_expert_train.h5
```

If you want a different dataset/cache location, edit `CACHE_DIR` in [`config.py`](/Users/matteo/Documents/le-wm/config.py).

## Training The World Model

The main training entrypoint is:

```bash
uv run python train.py
```

The training configuration is defined directly in [`config.py`](/Users/matteo/Documents/le-wm/config.py). The most relevant settings are:

- `USE_LEARNED_ACTIONS = True`
- `NUM_CODES = 8`
- `HISTORY_SIZE = 3`
- `NUM_PREDS = 1`
- `EMBED_DIM = 192`
- `ACTION_DIM = 2`
- `SIGREG_WEIGHT = 0.09`

Each training run writes a timestamped directory under:

```bash
runs/
```

with artifacts such as:

- `config.json`
- `metrics.tsv`
- `metrics.jsonl`
- `training_curves.png`
- `*_object.ckpt`
- `*_weights.ckpt`

## Training The Visual Decoder

The latent world model does not directly decode pixels during joint training. To render imagined rollouts, train a separate decoder afterwards:

```bash
uv run python scripts/train_decoder.py
```

Before running this, set `DECODER_SOURCE_CHECKPOINT` in [`config.py`](/Users/matteo/Documents/le-wm/config.py) to a trained world-model checkpoint.

The decoder is trained from latent states to pixels using the decoder-specific settings in `config.py`, including the top-k reconstruction loss.

## Training The PushT Action Translator

For PushT evaluation, latent actions can be translated back into executable continuous actions:

```bash
uv run python scripts/train_action_translator.py
```

Before running this, set `TRANSLATOR_SOURCE_CHECKPOINT` in [`config.py`](/Users/matteo/Documents/le-wm/config.py) to the trained world-model checkpoint you want to use.

This translator is specific to the control/evaluation setup where:

- the learned action space is discrete
- PushT expects continuous actions
- each latent action effectively summarizes multiple environment frames because the dataset uses frame skipping

## Evaluation And Planning

The evaluation entrypoint is:

```bash
uv run python eval.py
```

Important evaluation settings live in [`config.py`](/Users/matteo/Documents/le-wm/config.py), especially:

- `EVAL_POLICY`
- `EVAL_USE_ACTION_TRANSLATOR`
- `EVAL_TRANSLATOR_CHECKPOINT`
- planning horizon / solver settings

The default setup uses CEM planning over latent action sequences.

## Rendering Imagined Rollouts

After evaluation, you can decode saved planning latents into videos with:

```bash
uv run python scripts/render_eval_imagined_trajectories.py --input runs/
```

If needed, point it at a specific decoder checkpoint with `--decoder-checkpoint`.

## How This Differs From LeWorldModel

Compared with the original LeWorldModel repository/setup:

- real environment actions are not required for world-model training
- latent actions are discovered from passive video
- the latent-action module and world model train jointly
- the action bottleneck is discrete and codebook-based
- visual decoding is trained separately after the world model

Compared with multi-stage latent-action pipelines:

- there is no separate pixel-space latent-action pretraining stage
- the latent action model operates directly in world-model latent space
- the action vocabulary is shaped by what helps prediction and planning

## Notes

- The repo name and some internal filenames still use `le-wm` / `lewm`, but this repository now reflects **la leWorldModel**.
- The strongest supported workflow in the current code is PushT.
- Some config defaults point to machine-specific absolute checkpoint paths; update those in [`config.py`](/Users/matteo/Documents/le-wm/config.py) before running training or evaluation locally.

## Citation

If you use this code, cite the relevant LeWorldModel paper and also reference the `la leWorldModel` blog post that describes the latent-action extension implemented here.
