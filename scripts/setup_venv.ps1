$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDirectory = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
$requirementsPath = Join-Path $projectRoot 'requirements.txt'

if (-not (Test-Path -LiteralPath $venvPython)) {
    & python -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

& $venvPython -m pip install --disable-pip-version-check -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }

Write-Output 'Project virtual environment is ready.'
Write-Output "Python: $venvPython"
Write-Output 'Run scripts\verify_setup.py with that interpreter to verify CUDA and data loading.'
