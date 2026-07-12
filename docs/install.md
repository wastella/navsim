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
**macOS 12.3+** and an **Apple Silicon (M1 or later) Mac** — Intel Macs without a
discrete GPU can't use MPS and will silently fall back to CPU (still works, just slow).

- **Inference/eval**: no configuration needed. `AbstractAgent.compute_trajectory`
  automatically moves the model and features to `mps` when
  `torch.backends.mps.is_available()`, falling back to CPU otherwise.
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically at import time, so any op
  without an MPS kernel in this fork's pinned `torch==2.0.1` transparently runs on
  CPU instead of crashing (this only affects unimplemented ops, never changes
  results — just unset the env var yourself before running if you'd rather see a
  hard error than a silent CPU fallback while debugging).
- **Training**: `default_training.yaml` still defaults to `accelerator: gpu`,
  `strategy: ddp`, `precision: 16-mixed`, which are CUDA-only (`ddp` needs multiple
  CUDA devices; `16-mixed` AMP silently disables loss scaling on MPS, risking NaNs;
  `pin_memory` is a CUDA-only optimization and is a no-op elsewhere). Use the
  `*_macos.sh` variants instead of the standard training scripts, which override
  these to MPS-compatible settings (`strategy=auto`, `precision=32-true`,
  `pin_memory=false`):
  ```bash
  ./scripts/training/run_transfuser_training_macos.sh
  ./scripts/training/run_ego_mlp_agent_training_macos.sh
  ```
  **Note:** training in full `32-true` precision on Mac is not a like-for-like
  comparison with `16-mixed` CUDA runs — don't directly compare metrics between a
  Mac-trained checkpoint and the original CUDA/paper baselines without accounting
  for this. (`bf16-mixed` is not a usable middle ground here: it isn't supported by
  the MPS backend in `torch==2.0.1`.)
- **`guppy3`**: this dependency is skipped on macOS (`sys_platform != "darwin"`
  marker in `requirements.txt`) — it isn't used anywhere in `navsim/` and has known
  build issues on macOS.

#### Troubleshooting

- **`NotImplementedError: The operator 'aten::...' is not currently implemented for
  the MPS device`**: this shouldn't happen given the automatic CPU fallback above —
  if you see it anyway, confirm `PYTORCH_ENABLE_MPS_FALLBACK` isn't set to `0` in
  your shell.
- **`torch.backends.mps.is_available()` returns `False` on an Apple Silicon Mac**:
  usually means Python is running under Rosetta (x86_64) instead of natively
  (arm64). Check with `python -c "import platform; print(platform.machine())"` —
  it should print `arm64`; if it prints `x86_64`, reinstall a native arm64
  conda/Python.
- **`pip install -e .` fails trying to build `Fiona`, `rasterio`, or `geopandas`
  from source** instead of using a prebuilt wheel: install Xcode Command Line Tools
  (`xcode-select --install`) and retry. This shouldn't normally be needed — modern
  wheels for these packages bundle their own GDAL/GEOS/PROJ on macOS arm64 — but
  it's the standard fallback if pip can't find a matching wheel for your exact
  Python version.

This is a fork of the upstream [autonomousvision/navsim](https://github.com/autonomousvision/navsim)
devkit; see that repo for the original CUDA/Linux-focused installation path.
