#!/usr/bin/env bash
# check_macos_setup.sh
#
# "Doctor" script for the macOS / Apple Silicon fork of navsim. Run this before sinking
# hours into a training/eval run to sanity-check your machine and shell environment in one
# shot, instead of discovering problems one cryptic traceback at a time.
#
# Usage:
#   ./scripts/check_macos_setup.sh
#
# Exit status: 0 if there are no FAIL results (WARN is non-fatal), 1 otherwise.
#
# This script is read-only: it does not install anything, does not modify your shell
# config, and does not touch any files outside of /tmp.

set +e  # we want to run every check even if earlier ones fail

PASS=0
WARN=0
FAIL=0

pass() { printf "  [PASS] %s\n" "$1"; PASS=$((PASS + 1)); }
warn() { printf "  [WARN] %s\n" "$1"; WARN=$((WARN + 1)); }
fail() { printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL + 1)); }

section() { printf "\n== %s ==\n" "$1"; }

# ---------------------------------------------------------------------------
section "Operating system"
# ---------------------------------------------------------------------------

OS_NAME="$(uname -s)"
if [ "$OS_NAME" = "Darwin" ]; then
    pass "Running on macOS (uname -s == Darwin)"
else
    fail "Not running on macOS (uname -s == $OS_NAME) -- this script is only meaningful on macOS"
fi

if [ "$OS_NAME" = "Darwin" ] && command -v sw_vers >/dev/null 2>&1; then
    MACOS_VERSION="$(sw_vers -productVersion)"
    MACOS_MAJOR="$(echo "$MACOS_VERSION" | cut -d. -f1)"
    MACOS_MINOR="$(echo "$MACOS_VERSION" | cut -d. -f2)"
    # MPS requires macOS 12.3+
    if [ "$MACOS_MAJOR" -gt 12 ] || { [ "$MACOS_MAJOR" -eq 12 ] && [ "$MACOS_MINOR" -ge 3 ]; }; then
        pass "macOS version $MACOS_VERSION (>= 12.3, required for MPS)"
    else
        fail "macOS version $MACOS_VERSION is older than 12.3 -- MPS will not be available, training/eval will silently run on CPU only"
    fi
fi

# ---------------------------------------------------------------------------
section "CPU architecture"
# ---------------------------------------------------------------------------

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    pass "Native arm64 process (Apple Silicon, running natively)"
elif [ "$ARCH" = "x86_64" ]; then
    # Could be a genuine Intel Mac, or an Apple Silicon Mac running an x86_64
    # (Rosetta-translated) Python. Try to disambiguate using sysctl.
    if command -v sysctl >/dev/null 2>&1 && sysctl -n sysctl.proc_translated >/dev/null 2>&1; then
        TRANSLATED="$(sysctl -n sysctl.proc_translated 2>/dev/null)"
        if [ "$TRANSLATED" = "1" ]; then
            fail "Process is x86_64 under Rosetta on an Apple Silicon Mac (sysctl.proc_translated=1) -- torch.backends.mps.is_available() will report False. Reinstall a native arm64 conda/Python (e.g. via the arm64 Miniforge/Miniconda installer) and recreate your environment."
        else
            warn "Process architecture is x86_64 on genuine Intel hardware (not translated) -- there is no Apple GPU here, so MPS is unavailable. Inference/eval will fall back to CPU automatically (slow but works); running a training script with platform=mps sets trainer.params.accelerator=mps and will hard-fail with a MisconfigurationException on this machine. Omit platform=mps and let it use the default (CUDA-oriented) settings, overriding trainer.params.accelerator=cpu directly if you have no GPU at all."
        fi
    else
        warn "Process architecture is x86_64 and could not determine if this is Rosetta translation or genuine Intel hardware"
    fi
else
    warn "Unrecognized architecture: $ARCH"
fi

# ---------------------------------------------------------------------------
section "Python / environment variables"
# ---------------------------------------------------------------------------

for var in NUPLAN_MAP_VERSION NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT NAVSIM_DEVKIT_ROOT OPENSCENE_DATA_ROOT; do
    value="${!var}"
    if [ -z "$value" ]; then
        fail "\$$var is not set -- source env.sh (or add the exports to ~/.zshrc and open a new terminal) before running any navsim script. Symptom if you skip this: training scripts fail immediately with a bare 'No such file or directory', and eval/scoring scripts fail much later with a deep, unfamiliar-looking omegaconf/antlr traceback ending in \"Environment variable '$var' not found\"."
        continue
    fi
    case "$var" in
        NUPLAN_MAPS_ROOT|NAVSIM_EXP_ROOT|NAVSIM_DEVKIT_ROOT|OPENSCENE_DATA_ROOT)
            if [ -d "$value" ]; then
                pass "\$$var=$value (exists)"
            else
                fail "\$$var=$value is set but that directory does not exist"
            fi
            ;;
        *)
            pass "\$$var=$value"
            ;;
    esac
done

# ---------------------------------------------------------------------------
section "Python interpreter"
# ---------------------------------------------------------------------------

PYBIN=""
if command -v python >/dev/null 2>&1; then
    PYBIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYBIN="python3"
fi

if [ -z "$PYBIN" ]; then
    fail "No 'python' or 'python3' found on PATH -- activate your conda env (conda activate navsim) or venv first"
else
    PY_VERSION="$($PYBIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)"
    PY_ARCH="$($PYBIN -c 'import platform; print(platform.machine())' 2>/dev/null)"
    pass "Using '$PYBIN' -> $($PYBIN -c 'import sys; print(sys.executable)' 2>/dev/null) (Python $PY_VERSION, $PY_ARCH)"
    if [ "$ARCH" = "arm64" ] && [ "$PY_ARCH" != "arm64" ]; then
        fail "Shell is arm64 but Python reports machine()=='$PY_ARCH' -- this Python was installed under Rosetta. torch.backends.mps.is_available() will be False. Reinstall a native arm64 Python/conda."
    fi
fi

# ---------------------------------------------------------------------------
section "torch / pytorch-lightning / MPS"
# ---------------------------------------------------------------------------

if [ -n "$PYBIN" ]; then
    TORCH_CHECK="$($PYBIN - <<'EOF' 2>&1
import sys
try:
    import torch
except Exception as e:
    print("IMPORT_ERROR: " + repr(e))
    sys.exit(0)

pinned = "2.0.1"
print("TORCH_VERSION: " + torch.__version__)
print("TORCH_VERSION_MATCH: " + ("yes" if torch.__version__.split("+")[0] == pinned else "no"))
try:
    print("MPS_BUILT: " + str(torch.backends.mps.is_built()))
    print("MPS_AVAILABLE: " + str(torch.backends.mps.is_available()))
except Exception as e:
    print("MPS_CHECK_ERROR: " + repr(e))

try:
    import pytorch_lightning as pl
    print("LIGHTNING_VERSION: " + pl.__version__)
except Exception as e:
    print("LIGHTNING_IMPORT_ERROR: " + repr(e))
EOF
)"
    if echo "$TORCH_CHECK" | grep -q "^IMPORT_ERROR:"; then
        fail "Could not import torch in '$PYBIN' ($(echo "$TORCH_CHECK" | grep '^IMPORT_ERROR:' | sed 's/^IMPORT_ERROR: //')) -- has 'pip install -e .' been run in the active environment?"
    else
        TORCH_VERSION="$(echo "$TORCH_CHECK" | sed -n 's/^TORCH_VERSION: //p')"
        VERSION_MATCH="$(echo "$TORCH_CHECK" | sed -n 's/^TORCH_VERSION_MATCH: //p')"
        MPS_BUILT="$(echo "$TORCH_CHECK" | sed -n 's/^MPS_BUILT: //p')"
        MPS_AVAILABLE="$(echo "$TORCH_CHECK" | sed -n 's/^MPS_AVAILABLE: //p')"
        LIGHTNING_VERSION="$(echo "$TORCH_CHECK" | sed -n 's/^LIGHTNING_VERSION: //p')"

        if [ "$VERSION_MATCH" = "yes" ]; then
            pass "torch==$TORCH_VERSION (matches the version this fork is pinned/tested against)"
        else
            warn "torch==$TORCH_VERSION does not match the pinned 2.0.1 -- this fork has no runtime version check/guard, so a mismatch will NOT fail fast. Known risks: bf16-mixed precision claims, the PYTORCH_ENABLE_MPS_FALLBACK behavior, and MPS op-coverage described in docs/install.md are only verified against 2.0.1. If you hit an unexplained error, first try 'pip install torch==2.0.1 torchvision==0.15.2'."
        fi

        if [ "$MPS_BUILT" = "True" ]; then
            pass "torch was built with MPS support (torch.backends.mps.is_built() == True)"
        else
            fail "torch was NOT built with MPS support -- likely an x86_64/Rosetta or non-macOS wheel"
        fi

        if [ "$MPS_AVAILABLE" = "True" ]; then
            pass "MPS device is available (torch.backends.mps.is_available() == True) -- inference/eval and training runs invoked with platform=mps will use the GPU"
        else
            warn "MPS device is NOT available (torch.backends.mps.is_available() == False). Inference/eval (AbstractAgent.compute_trajectory) will silently fall back to CPU -- slower but fine. Training invoked with platform=mps hardcodes trainer.params.accelerator=mps and will hard-fail with a pytorch_lightning MisconfigurationException here -- omit platform=mps (or override trainer.params.accelerator=cpu directly) instead."
        fi

        if [ -n "$LIGHTNING_VERSION" ]; then
            pass "pytorch-lightning==$LIGHTNING_VERSION importable"
        else
            fail "pytorch-lightning is not importable -- $(echo "$TORCH_CHECK" | grep '^LIGHTNING_IMPORT_ERROR:')"
        fi
    fi
fi

# ---------------------------------------------------------------------------
section "PYTORCH_ENABLE_MPS_FALLBACK"
# ---------------------------------------------------------------------------

if [ "${PYTORCH_ENABLE_MPS_FALLBACK}" = "0" ]; then
    warn "\$PYTORCH_ENABLE_MPS_FALLBACK is explicitly set to 0 -- ops without an MPS kernel will hard-error (torch's NotImplementedError) instead of silently falling back to CPU. This may be intentional if you're debugging; unset it otherwise."
else
    pass "\$PYTORCH_ENABLE_MPS_FALLBACK is unset or non-zero -- navsim/agents/abstract_agent.py also sets this to 1 automatically at import time regardless, so this is only informational"
fi

# ---------------------------------------------------------------------------
section "NAVSIM_DEVICE (inference device override)"
# ---------------------------------------------------------------------------

if [ -n "${NAVSIM_DEVICE}" ]; then
    warn "\$NAVSIM_DEVICE=$NAVSIM_DEVICE -- this forces AbstractAgent.compute_trajectory's inference device, overriding the normal cuda > mps > cpu auto-detection. Unset it unless you're deliberately forcing a specific device (e.g. for debugging)."
else
    pass "\$NAVSIM_DEVICE is unset -- inference will auto-detect cuda > mps > cpu"
fi

# ---------------------------------------------------------------------------
section "Summary"
# ---------------------------------------------------------------------------

printf "\n%d passed, %d warning(s), %d failure(s)\n" "$PASS" "$WARN" "$FAIL"

if [ "$FAIL" -gt 0 ]; then
    printf "\nYour setup has issues that will very likely break a training/eval run. Fix the [FAIL] items above first.\n"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    printf "\nNo hard failures, but review the [WARN] items above -- some may still bite you depending on which scripts you run.\n"
    exit 0
else
    printf "\nEverything checks out.\n"
    exit 0
fi
