$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonCandidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "venv\Scripts\python.exe")
)
$Python = $PythonCandidates |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1

if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Install Python 3.11+ and add it to PATH."
    }
    $Python = $PythonCommand.Source
}

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed for '$Python'. Run: $Python -m pip install -e '.[build]'"
}

$WorkPath = Join-Path $Root "build\windows"
$DistPath = Join-Path $Root "releases\windows"

Push-Location $Root
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --workpath $WorkPath `
        --distpath $DistPath `
        (Join-Path $Root "packaging\boltzmann.spec")

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$Executable = Join-Path $DistPath "BoltzmannLab.exe"
if (-not (Test-Path $Executable)) {
    throw "Build completed without producing the expected executable: $Executable"
}

Write-Host "Windows build completed: $Executable"
