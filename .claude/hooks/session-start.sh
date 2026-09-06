#!/bin/bash
#
# SessionStart hook — prepares a Claude Code on the web container for this repo.
#
# The container is ephemeral: it is rebuilt from the image on every session, so
# anything pip installed by hand vanishes. This script restores the pieces a
# session actually needs.
#
# What it deliberately does NOT install: torch, segmentation-models-pytorch and
# albumentations. Training never happens in this container — it happens in a
# Kaggle kernel with a T4 attached. Installing torch here would cost gigabytes
# of the session's fixed disk allowance and several minutes of startup to
# produce a GPU stack with no GPU under it. `shared/utils.py` imports torch
# lazily inside a function precisely so the unit tests run without it.
#
# Set SOL_INSTALL_TORCH=1 to pull the full requirements.txt anyway, for the rare
# session that wants to execute a training step locally on CPU.
#
set -euo pipefail

# Local machines already have their own environments; only the web container
# needs rebuilding from scratch.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "session-start: not a remote session, nothing to do"
  exit 0
fi

REPO_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

echo "session-start: preparing $(basename "$REPO_DIR")"

# --------------------------------------------------------------------------- #
# 1. Python dependencies
# --------------------------------------------------------------------------- #
# opencv-python-headless rather than opencv-python: the container has no X11,
# and the GUI build fails to import with a missing libGL.so.1.
PACKAGES=(
  "kaggle>=1.6"           # push kernels, poll runs, fetch output, submit
  "numpy>=1.24"
  "pandas>=2.0"
  "pycocotools>=2.0.7"    # RLE encode/decode for submissions
  "PyYAML>=6.0"           # config.yaml
  "opencv-python-headless>=4.8"
  "pytest>=7.0"           # tests/test_utils.py
)

# No --upgrade: several of these (PyYAML, numpy) ship as distro-managed
# packages, and upgrading them makes pip try to uninstall a package whose
# RECORD file it does not own, which fails the whole hook. Without the flag pip
# installs only what is missing or below the floor, which is what is wanted and
# is faster on a warm container.
echo "session-start: installing control-plane packages"
python3 -m pip install --quiet --root-user-action=ignore "${PACKAGES[@]}"

if [ "${SOL_INSTALL_TORCH:-}" = "1" ]; then
  echo "session-start: SOL_INSTALL_TORCH=1, installing full requirements.txt"
  python3 -m pip install --quiet --root-user-action=ignore -r requirements.txt
fi

# --------------------------------------------------------------------------- #
# 2. Kaggle credentials
# --------------------------------------------------------------------------- #
# The CLI reads KAGGLE_USERNAME/KAGGLE_KEY directly, so the env vars alone are
# enough. kaggle.json is written as well because some code paths and the
# `kaggle config` subcommands look for the file rather than the environment.
# It is written to $HOME, never into the repo, and $HOME is not version
# controlled — but .gitignore lists kaggle.json regardless, as a second guard.
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  mkdir -p "$HOME/.kaggle"
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    printf '{"username":"%s","key":"%s"}\n' \
      "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$HOME/.kaggle/kaggle.json"
  fi
  # The CLI refuses to run against a world-readable credential file.
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "session-start: kaggle credentials present for user '$KAGGLE_USERNAME'"
else
  echo "session-start: WARNING - KAGGLE_USERNAME/KAGGLE_KEY not set;" \
       "kernel pushes and submissions will fail until they are"
fi

# --------------------------------------------------------------------------- #
# 3. Session environment
# --------------------------------------------------------------------------- #
# `from shared.utils import ...` only resolves if the repo root is importable.
# pytest puts tests/ on sys.path, not the root, so without this the test suite
# fails at collection with ModuleNotFoundError: shared.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PYTHONPATH=\"$REPO_DIR\${PYTHONPATH:+:\$PYTHONPATH}\"" >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: ready"
