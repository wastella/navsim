# Download and installation

To get started with NAVSIM:

### 1. Clone the navsim-devkit

Clone the repository

```bash
git clone https://github.com/wastella/navsim.git
cd navsim
```

(This is a fork of the upstream [autonomousvision/navsim](https://github.com/autonomousvision/navsim)
devkit that adds macOS / Apple Silicon support — see the [macOS / Apple Silicon](#macos--apple-silicon)
section below. If you specifically want the original CUDA/Linux-only devkit, clone
`autonomousvision/navsim` instead.)

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.
You can download these splits with the following scripts.

```bash
./download_mini
./download_trainval
./download_test
./download_warmup_two_stage
./download_navhard_two_stage
./download_private_test_hard_two_stage
```

Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split.

This will download the splits into the download directory. From there, move it to create the following structure.

```angular2html
~/navsim_workspace
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         └── private_test_hard.pkl
    │    └── mini
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         ├──  CAM_B0
    |    |         ├──  CAM_F0
    |    |         ├──   ...
    |    └── mini
    └── navhard_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── warmup_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── private_test_hard_two_stage
         ├── openscene_meta_datas
         └── sensor_blobs

```
Set the required environment variables, by adding the following to your `~/.bashrc` file
(on macOS, the default shell is zsh — use `~/.zshrc` instead).
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

Then run `source ~/.bashrc` (or `source ~/.zshrc` on macOS) — or just open a new terminal —
so these variables take effect. They must be set in every shell you use to run navsim;
forgetting this step is the most common cause of confusing early failures (an immediate
`No such file or directory` from training scripts, or a much longer, unfamiliar-looking
`omegaconf`/`antlr` traceback from eval/scoring scripts, both ultimately because one of
these variables was empty).

⏰ **Note:** The `navhard_two_stage` split is used for local testing of your model's performance in a two-stage pseudo closed-loop setup.
In contrast, `warmup_two_stage` is a smaller dataset designed for validating and testing submissions to the [Hugging Face Warmup leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup).
In other words, the results you obtain locally on `warmup_two_stage` should match the results you see after submitting to Hugging Face.
`private_test_hard_two_stage` contains the challenge data.
You will need it to generate a `submission.pkl` in order to participate in the official challenge on the [Hugging Face CPVR 2025 leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-internal) (for more details, see [Submission](submission.md)).

### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

### macOS / Apple Silicon

This fork adds support for running NAVSIM on Macs with an Apple GPU, using PyTorch's
Metal Performance Shaders (MPS) backend instead of CPU-only inference. Requires
**macOS 12.3+** and an **Apple Silicon (M1 or later) Mac** for MPS. Intel Macs (no
Apple GPU) can still run this fork, just without GPU acceleration — see the
inference/training split below, since the two behave differently there.

**Before anything else**, run `./scripts/check_macos_setup.sh` — it checks your macOS
version, chip architecture (including Rosetta mismatches), env vars, torch/MPS/
pytorch-lightning versions, and reports pass/warn/fail for each in one shot, instead of
you discovering the same issues one cryptic traceback at a time over a multi-hour run.

There are two independent device-selection mechanisms in this fork — inference and
training don't share code, so fixing/configuring one doesn't affect the other:

- **Inference/eval** (`AbstractAgent.compute_trajectory`, used by all eval/submission
  scripts): no configuration needed. Auto-detects in priority order **CUDA, then MPS,
  then CPU** — so this is also safe to use unchanged on non-Mac machines. Set
  `NAVSIM_DEVICE=cpu` (or `cuda`/`mps`) to force a specific device, e.g. for debugging.
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically at import time, so any op
  without an MPS kernel in this fork's pinned `torch==2.0.1` transparently runs on
  CPU instead of crashing (this only affects unimplemented ops, never changes
  results — unset the env var yourself before running if you'd rather see a hard
  error than a silent CPU fallback while debugging). On a Mac with no Apple GPU
  (older Intel Macs), this path **does** fall back to CPU silently — slow, but it works.
- **Training**: the default training config (`trainer.params.accelerator: gpu`,
  `strategy: ddp`, `precision: 16-mixed`) is CUDA-only and **does not auto-detect** —
  running it on a Mac fails outright (`ddp` needs multiple CUDA devices). Append
  `platform=mps` to any training script to switch to MPS-compatible settings:
  ```bash
  ./scripts/training/run_transfuser_training.sh platform=mps
  ./scripts/training/run_ego_mlp_agent_training.sh platform=mps
  ```
  This is a genuinely different code path from inference, so **`platform=mps` hard-fails
  with a `pytorch_lightning.MisconfigurationException` on a Mac with no Apple GPU** —
  it does not silently fall back like the inference path does. On an Intel Mac, omit
  `platform=mps` and instead override `trainer.params.accelerator=cpu` directly.

  `platform=mps` (`navsim/planning/script/config/training/platform/mps.yaml`) sets:
  - `strategy=auto` (`ddp` needs multiple CUDA devices; `auto` resolves correctly to a
    single-device strategy for one MPS GPU)
  - `precision=32-true` (`16-mixed` AMP silently disables loss scaling on MPS, risking
    NaNs; `bf16-mixed` isn't supported by the MPS backend on this fork's pinned
    `torch==2.0.1`, so full precision is the safe choice)
  - `pin_memory=false` (a CUDA-only optimization, a no-op elsewhere)
  - `batch_size=16` and `num_workers=2`, down from the CUDA defaults (64 / 4): Apple
    Silicon uses unified memory shared with the OS rather than a discrete GPU's own
    VRAM, so a CUDA-sized batch risks swapping/system-wide slowdown instead of a clean
    OOM error. Tune based on your Mac's RAM — this is a conservative starting point,
    not a hard limit. Fewer, larger `num_workers` also amortize a real (if modest)
    macOS-specific cost better: macOS's default multiprocessing start method is
    `spawn`, not Linux's `fork`, so every DataLoader worker re-imports navsim's full
    dependency chain from scratch on startup instead of inheriting it.

  **Note:** training in full `32-true` precision on Mac is not a like-for-like
  comparison with `16-mixed` CUDA runs — don't directly compare metrics between a
  Mac-trained checkpoint and the original CUDA/paper baselines without accounting
  for this.
- **`guppy3`**: this dependency is skipped on macOS (`sys_platform != "darwin"`
  marker in `requirements.txt`) — it isn't used anywhere in `navsim/` and has known
  build issues on macOS.

#### Running long, unattended jobs on a laptop

Unlike a managed cluster, a personal Mac has no job scheduler or preemption handling —
a few things that are automatic elsewhere need to be done manually here:

- **Prevent sleep**: wrap long runs with `caffeinate -i`, e.g.
  `caffeinate -i ./scripts/training/run_transfuser_training.sh platform=mps` — otherwise
  macOS can suspend the process mid-epoch if the lid closes or the display sleeps.
- **Survive closing the terminal**: run under `tmux`/`screen`, or `nohup ... &`, so an
  SSH drop or closed terminal window doesn't kill the job.
- **Resuming a killed run**: pytorch-lightning's `Trainer` checkpoints by default under
  `<output_dir>/checkpoints`; pass `ckpt_path=<path to .ckpt>` to resume from the last
  completed epoch instead of starting over.
- **Watching progress**: this fork's `requirements.txt` already includes `tensorboard`;
  run `tensorboard --logdir <output_dir>` to watch loss/throughput. This is also the
  easiest way to notice if training is silently running large chunks on CPU (via the
  MPS fallback above) instead of the GPU — throughput will look far lower than expected.

#### Troubleshooting

- **`NotImplementedError: The operator 'aten::...' is not currently implemented for
  the MPS device`**: this shouldn't happen given the automatic CPU fallback above —
  if you see it anyway, confirm `PYTORCH_ENABLE_MPS_FALLBACK` isn't set to `0` in
  your shell.
- **`torch.backends.mps.is_available()` returns `False` on an Apple Silicon Mac**:
  usually means Python is running under Rosetta (x86_64) instead of natively
  (arm64). Check with `python -c "import platform; print(platform.machine())"` —
  it should print `arm64`; if it prints `x86_64`, reinstall a native arm64
  conda/Python. `./scripts/check_macos_setup.sh` checks this for you automatically.
- **`pip install -e .` fails trying to build `Fiona`, `rasterio`, or `geopandas`
  from source** instead of using a prebuilt wheel: install Xcode Command Line Tools
  (`xcode-select --install`) and retry. This shouldn't normally be needed — modern
  wheels for these packages bundle their own GDAL/GEOS/PROJ on macOS arm64 — but
  it's the standard fallback if pip can't find a matching wheel for your exact
  Python version.
- **Still stuck?** Run `./scripts/check_macos_setup.sh` first — it catches the most
  common setup issues (wrong architecture, missing env vars, torch/MPS mismatches)
  in one pass before you sink hours into a run that was never going to work.

This is a fork of the upstream [autonomousvision/navsim](https://github.com/autonomousvision/navsim)
devkit; see that repo for the original CUDA/Linux-focused installation path.
