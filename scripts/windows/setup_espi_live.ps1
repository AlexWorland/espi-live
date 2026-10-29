[CmdletBinding()]
param(
    [string[]]$AppArguments = @()
)

# Preflight and launcher for the maintained ESPI application.  This script is
# intentionally interactive: it never installs Python packages or the SDK
# without a clear yes/no answer, and it launches the application only when a
# fresh run finds every dependency healthy.

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$srcPath = Join-Path $repoRoot 'src'
$appPath = Join-Path $srcPath 'espi_live\__main__.py'
$wheelPath = Join-Path $repoRoot 'third_party\flir-spinnaker-4.2\python-wheel\spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl'
$sdkInstaller = Join-Path $PSScriptRoot 'install_spinnaker_sdk.bat'

function Write-Status {
    param([string]$Message)
    Write-Host "[ESPI setup] $Message"
}

function Confirm-Action {
    param([string]$Question)
    $answer = Read-Host "$Question [y/N]"
    return $answer -match '^(y|yes)$'
}

function Find-Python310 {
    # Prefer the interpreter installed alongside PySpin by the Spinnaker SDK.
    $candidates = @(
        (Join-Path $env:SystemDrive 'Python310\PySpin\python.exe'),
        (Join-Path $env:SystemDrive 'Python310\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return @{ Command = $candidate; Prefix = @() }
        }
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source '-3.10' '--version' *> $null
        if ($LASTEXITCODE -eq 0) {
            return @{ Command = $launcher.Source; Prefix = @('-3.10') }
        }
    }
    return $null
}

function Invoke-SelectedPython {
    param(
        [string[]]$Arguments,
        [switch]$Quiet
    )

    # A failed import is the normal signal that a dependency is missing. Keep
    # PowerShell from converting Python's traceback into a terminating native
    # command error before the caller can inspect the process exit code.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($Quiet) {
            & $script:python.Command @($script:python.Prefix) @Arguments 1>$null 2>$null
        } else {
            & $script:python.Command @($script:python.Prefix) @Arguments
        }
        $script:lastPythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

if (-not (Test-Path $appPath)) {
    throw "The ESPI application was not found: $appPath"
}

$python = Find-Python310
if (-not $python) {
    Write-Status 'Python 3.10 x64 is required but was not found.'
    if ((Get-Command winget.exe -ErrorAction SilentlyContinue) -and
        (Confirm-Action 'Install Python 3.10 x64 with winget now?')) {
        & winget.exe install --id Python.Python.3.10 --exact --source winget
        if ($LASTEXITCODE -ne 0) {
            throw 'Python installation did not complete successfully.'
        }
        Write-Status 'Python was installed. Run this helper again to continue setup.'
        exit 2
    }
    Write-Status 'Install Python 3.10 x64, then run this helper again.'
    exit 2
}

Write-Status "Using Python: $($python.Command) $($python.Prefix -join ' ')"

$moduleChecks = @{
    'opencv-python' = 'cv2'
    'numpy'          = 'numpy'
    'PySide6'        = 'PySide6.QtWidgets'
    'cv2_enumerate_cameras' = 'cv2_enumerate_cameras'
    'PySpin'         = 'PySpin'
}
$missing = [System.Collections.Generic.List[string]]::new()
foreach ($displayName in $moduleChecks.Keys) {
    $moduleName = $moduleChecks[$displayName]
    Invoke-SelectedPython -Arguments @('-c', "import $moduleName") -Quiet
    if ($lastPythonExitCode -ne 0) {
        $missing.Add($displayName)
    }
}

if ($missing.Count -eq 0) {
    Write-Status 'All dependencies are valid. Starting espi_live.py.'
    $env:PYTHONPATH = $srcPath
    Invoke-SelectedPython (@('-m', 'espi_live') + $AppArguments)
    exit $lastPythonExitCode
}

Write-Status "Missing or unusable dependencies: $($missing -join ', ')"

if ($missing -contains 'PySpin') {
    if (-not (Test-Path $sdkInstaller)) {
        throw "The bundled Spinnaker SDK installer helper was not found: $sdkInstaller"
    }
    if (Confirm-Action 'PySpin needs the FLIR Spinnaker SDK. Launch the bundled SDK installer now?') {
        & $sdkInstaller
        Write-Status 'The SDK installer closed. Run this helper again so it can check PySpin and install the wheel.'
        exit 2
    }
}

$pipPackages = @()
if ($missing -contains 'opencv-python') { $pipPackages += 'opencv-python' }
if ($missing -contains 'numpy') { $pipPackages += 'numpy' }
if ($missing -contains 'PySide6') { $pipPackages += 'PySide6' }
if ($missing -contains 'cv2_enumerate_cameras') { $pipPackages += 'cv2_enumerate_cameras' }

if ($pipPackages.Count -gt 0) {
    if (Confirm-Action "Install or repair: $($pipPackages -join ', ')?") {
        Invoke-SelectedPython @('-m', 'ensurepip', '--upgrade') | Out-Host
        if ($lastPythonExitCode -ne 0) { throw 'Could not prepare pip.' }
        Invoke-SelectedPython (@('-m', 'pip', 'install', '--upgrade') + $pipPackages) | Out-Host
        if ($lastPythonExitCode -ne 0) { throw 'Package installation failed.' }
    }
}

if (($missing -contains 'PySpin') -and (Test-Path $wheelPath)) {
    if (Confirm-Action 'Install the bundled PySpin wheel for Python 3.10 now?') {
        Invoke-SelectedPython @('-m', 'pip', 'install', '--upgrade', $wheelPath) | Out-Host
        if ($lastPythonExitCode -ne 0) { throw 'PySpin wheel installation failed.' }
    }
} elseif ($missing -contains 'PySpin') {
    Write-Status "PySpin wheel not found: $wheelPath"
}

Write-Status 'Setup actions are complete. Run this helper again; it will launch the app only after every dependency check passes.'
exit 2
