
$ErrorActionPreference = "Stop"

$REPO = "https://raw.githubusercontent.com/BarnstedtLab/RodeOFlow/master"
$ENV_FILE = "environment_setup_windows.yaml"
$ENV_NAME = "rodeoflow"

Write-Host "======================================"
Write-Host "       RodeOFlow Windows Installer"
Write-Host "======================================"

# Check prerequisites
foreach ($cmd in @("conda", "git")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "$cmd is not installed or not available in PATH."
    }
}

conda --version
git --version

# Download environment
$TEMP_YAML = Join-Path $env:TEMP $ENV_FILE

Write-Host "Downloading Windows environment..."

Invoke-WebRequest `
    -Uri "$REPO/$ENV_FILE" `
    -OutFile $TEMP_YAML

try {
    # Create environment
    Write-Host "Creating Conda environment..."

    conda env create -f $TEMP_YAML

    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment creation failed."
    }

    # Install Lightning Pose dependencies
    Write-Host "Installing Lightning Pose dependencies..."

    conda run -n $ENV_NAME python -m pip install `
        "numpy<2" `
        fiftyone `
        hydra-core `
        imgaug `
        kaleido `
        kornia `
        lightning `
        matplotlib `
        "moviepy<2"

    if ($LASTEXITCODE -ne 0) {
        throw "Lightning Pose dependencies installation failed."
    }

    # Install DeepLabCut
    Write-Host "Installing DeepLabCut..."

    conda run -n $ENV_NAME python -m pip install --pre deeplabcut

    if ($LASTEXITCODE -ne 0) {
        throw "DeepLabCut installation failed."
    }

    # Install Lightning Pose without DALI
    Write-Host "Installing Lightning Pose 1.8.1..."

    conda run -n $ENV_NAME python -m pip install `
        lightning-pose==1.8.1 --no-deps

    if ($LASTEXITCODE -ne 0) {
        throw "Lightning Pose installation failed."
    }

    # Install RodeOFlow
    Write-Host "Installing RodeOFlow..."

    conda run -n $ENV_NAME python -m pip install `
        git+https://github.com/BarnstedtLab/RodeOFlow.git --no-deps

    if ($LASTEXITCODE -ne 0) {
        throw "RodeOFlow installation failed."
    }

    # Verify
    Write-Host "Verifying installation..."

    conda run -n $ENV_NAME python -c `
        "from rodeoflow import pose_face, analyse_face; print('RodeOFlow OK')"

    if ($LASTEXITCODE -ne 0) {
        throw "Installation verification failed."
    }

    Write-Host ""
    Write-Host "======================================"
    Write-Host "Installation completed successfully!"
    Write-Host "Activate with: conda activate $ENV_NAME"
    Write-Host "======================================"
}
finally {
    Remove-Item $TEMP_YAML -ErrorAction SilentlyContinue
}
