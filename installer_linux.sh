
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="https://raw.githubusercontent.com/BarnstedtLab/RodeOFlow/dev"
ENV_FILE="environment_setup_linux.yaml"
ENV_NAME="rodeoflow"

echo "======================================"
echo "       RodeOFlow Linux Installer"
echo "======================================"

# Check prerequisites
for cmd in conda git curl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd is not installed or not in PATH."
        exit 1
    fi
done

conda --version
git --version

# Download environment
TMP_YAML="$(mktemp -t rodeoflow_envXXXXXX).yaml"
trap 'rm -f "$TMP_YAML"' EXIT

echo "Downloading environment..."
curl -fsSL "$REPO/environment/$ENV_FILE" -o "$TMP_YAML"

# Create environment
echo "Creating Conda environment..."
conda env create -f "$TMP_YAML"

# Install DeepLabCut
echo "Installing DeepLabCut..."
conda run -n "$ENV_NAME" \
    python -m pip install --pre deeplabcut

# Install RodeOFlow
echo "Installing RodeOFlow..."
conda run -n "$ENV_NAME" \
    python -m pip install \
    git+https://github.com/BarnstedtLab/RodeOFlow.git@dev --no-deps

# Verify
echo "Verifying installation..."
conda run -n "$ENV_NAME" python -c \
    "from rodeoflow import pose_face, analyse_face; print('RodeOFlow OK')"

echo ""
echo "======================================"
echo "Installation completed successfully!"
echo "Activate with: conda activate $ENV_NAME"
echo "======================================"
