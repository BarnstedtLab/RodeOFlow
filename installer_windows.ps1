
$ErrorActionPreference = "Stop"

$REPO = "https://raw.githubusercontent.com/BarnstedtLab/RodeOFlow/dev"
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
    -Uri "$REPO/environment/$ENV_FILE" `
    -OutFile $TEMP_YAML

try {
    # Create environment
    Write-Host "Creating Conda environment..."

    conda env create -f $TEMP_YAML

    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment creation failed."
    }

    # Install DeepLabCut
    Write-Host "Installing DeepLabCut..."

    conda run -n $ENV_NAME python -m pip install --pre deeplabcut

    if ($LASTEXITCODE -ne 0) {
        throw "DeepLabCut installation failed."
    }

    # Install RodeOFlow
    Write-Host "Installing RodeOFlow..."

    conda run -n $ENV_NAME python -m pip install `
        git+https://github.com/BarnstedtLab/RodeOFlow.git@dev --no-deps

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
