
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="https://raw.githubusercontent.com/BarnstedtLab/RodeOFlow/master"
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
TMP_YAML="$(mktemp)"
trap 'rm -f "$TMP_YAML"' EXIT

echo "Downloading environment..."
curl -fsSL "$REPO/$ENV_FILE" -o "$TMP_YAML"

# Create environment
echo "Creating Conda environment..."
conda env create -f "$TMP_YAML"

# Install Lightning Pose dependencies (without DALI)
echo "Installing Lightning Pose dependencies..."
conda run -n "$ENV_NAME" python -m pip install \
    "numpy<2" \
    fiftyone \
    hydra-core \
    imgaug \
    kaleido \
    kornia \
    lightning \
    matplotlib \
    "moviepy<2"

# Install DeepLabCut
echo "Installing DeepLabCut..."
conda run -n "$ENV_NAME" \
    python -m pip install --pre deeplabcut

# Install Lightning Pose without NVIDIA DALI
echo "Installing Lightning Pose 1.8.1..."
conda run -n "$ENV_NAME" \
    python -m pip install lightning-pose==1.8.1 --no-deps

# Install RodeOFlow
echo "Installing RodeOFlow..."
conda run -n "$ENV_NAME" \
    python -m pip install \
    git+https://github.com/BarnstedtLab/RodeOFlow.git --no-deps

# Verify
echo "Verifying installation..."
conda run -n "$ENV_NAME" python -c \
    "from rodeoflow import pose_face, analyse_face; print('RodeOFlow OK')"

echo ""
echo "======================================"
echo "Installation completed successfully!"
echo "Activate with: conda activate $ENV_NAME"
echo "======================================"
