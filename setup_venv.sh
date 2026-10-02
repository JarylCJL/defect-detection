#!/usr/bin/env bash
# Bootstraps the pyenv-managed Python virtualenv for this repo on a fresh
# machine: installs pyenv itself if missing, installs the pinned Python
# version, creates the "defect-detection" virtualenv, and installs
# requirements.txt into it. It also downloads the YOLO-NAS-M COCO pretrained
# weights into the torch cache, because the address hardcoded in
# super-gradients 3.7.1 (sghub.deci.ai) no longer exists; with the file in
# the cache, train.ipynb's INIT_SOURCE="coco" works without any download.
#
# Usage: ./setup_venv.sh
#
# If pyenv was just installed for the first time, restart your shell (its
# shell hooks only take effect in a new session) and rerun this script.

set -euo pipefail

PYTHON_VERSION="3.10.19"
VENV_NAME="defect-detection"
# Current official host for the SuperGradients pretrained weights (the one
# used on super-gradients' master branch).
COCO_WEIGHTS_URL="https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_m_coco.pth"
COCO_WEIGHTS_FILE="yolo_nas_m_coco.pth"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Install pyenv (+ pyenv-virtualenv) if missing.
if ! command -v pyenv >/dev/null 2>&1; then
    echo "pyenv not found - installing (this also installs the pyenv-virtualenv plugin)..."
    curl -fsSL https://pyenv.run | bash

    SHELL_RC="${HOME}/.bashrc"
    [ -n "${ZSH_VERSION:-}" ] && SHELL_RC="${HOME}/.zshrc"

    if ! grep -q "PYENV_ROOT" "$SHELL_RC" 2>/dev/null; then
        {
            echo ''
            echo '# pyenv (added by setup_venv.sh)'
            echo 'export PYENV_ROOT="$HOME/.pyenv"'
            echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"'
            echo 'eval "$(pyenv init -)"'
            echo 'eval "$(pyenv virtualenv-init -)"'
        } >> "$SHELL_RC"
        echo "Added pyenv shell hooks to $SHELL_RC"
    fi

    echo
    echo "pyenv was just installed. Restart your shell (open a new terminal, or run 'exec \$SHELL') and rerun this script."
    exit 0
fi

# NOTE: pyenv's shell-integration commands (init/virtualenv-init) redefine
# `pyenv` as a shell function. Piping directly from a call to that function
# (e.g. `pyenv versions --bare | grep ...`) is unreliable under `set -e`/
# `pipefail` - it can report false negatives. Always capture to a variable
# first, then grep the variable, as done throughout below.

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
PYENV_ROOT="$(pyenv root)"

pyenv_commands="$(pyenv commands)"
if ! grep -qx "virtualenv" <<< "$pyenv_commands"; then
    echo "pyenv is installed but the pyenv-virtualenv plugin isn't." >&2
    echo "Install it: https://github.com/pyenv/pyenv-virtualenv#installation" >&2
    exit 1
fi

# 2. Install the pinned Python version if missing.
installed_versions="$(pyenv versions --bare)"
if ! grep -qx "$PYTHON_VERSION" <<< "$installed_versions"; then
    echo "Installing Python $PYTHON_VERSION via pyenv (compiles from source, a few minutes)..."
    pyenv install "$PYTHON_VERSION"
fi

# 3. Create the virtualenv if missing.
installed_versions="$(pyenv versions --bare)"
if ! grep -qx "${PYTHON_VERSION}/envs/${VENV_NAME}" <<< "$installed_versions"; then
    echo "Creating virtualenv '$VENV_NAME' (Python $PYTHON_VERSION)..."
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi

# 4. Install dependencies directly via the venv's own pip (no need to
# activate anything for this script to work).
VENV_PIP="${PYENV_ROOT}/versions/${VENV_NAME}/bin/pip"
echo "Installing requirements.txt into '$VENV_NAME'..."
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r "${REPO_ROOT}/requirements.txt"

# 5. Put the YOLO-NAS-M COCO weights where super-gradients looks before
# downloading (torch's hub cache). A failed download is not fatal: only
# training from COCO weights needs the file.
TORCH_CACHE_DIR="${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}/hub/checkpoints"
if [ -s "${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}" ]; then
    echo "COCO pretrained weights already present: ${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}"
else
    echo "Downloading YOLO-NAS-M COCO pretrained weights (~205 MB)..."
    mkdir -p "$TORCH_CACHE_DIR"
    if curl -fL --retry 3 -o "${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}.part" "$COCO_WEIGHTS_URL"; then
        mv "${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}.part" "${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}"
    else
        rm -f "${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}.part"
        echo "WARNING: could not download ${COCO_WEIGHTS_URL}" >&2
        echo "         train.ipynb with INIT_SOURCE=\"coco\" will fail until the file is at" >&2
        echo "         ${TORCH_CACHE_DIR}/${COCO_WEIGHTS_FILE}" >&2
    fi
fi

echo
echo "Done. cd into ${REPO_ROOT} (or any subdirectory) - .python-version will auto-activate '$VENV_NAME'."
echo "Verify with: pyenv version && python --version"
