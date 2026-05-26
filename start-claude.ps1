param(
    [Parameter(Mandatory=$false)]
    [string]$EnvName,

    [Parameter(Mandatory=$false)]
    [string]$Version = "2.0.49",

    [Parameter(Mandatory=$false)]
    [string]$WorkDir
)

# UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# Set working directory
if ($WorkDir) {
    Set-Location $WorkDir
}

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Restarting with admin privileges..." -ForegroundColor Yellow
    $currentDir = (Get-Location).Path
    $args = "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`" -WorkDir `"$currentDir`""
    if ($EnvName) { $args += " -EnvName `"$EnvName`"" }
    if ($Version -ne "2.0.49") { $args += " -Version `"$Version`"" }
    Start-Process powershell.exe -ArgumentList $args -Verb RunAs
    exit
}

Write-Host "=== Claude Code Launcher ===" -ForegroundColor Cyan
Write-Host "Working Directory: $(Get-Location)" -ForegroundColor Green

# Get environment name
if (-not $EnvName) {
    $EnvName = Read-Host "`nEnter Conda environment name"
    if (-not $EnvName) {
        Write-Host "Environment name is required." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "Environment: $EnvName" -ForegroundColor Green
Write-Host "Version: $Version" -ForegroundColor Green

# Find Conda
Write-Host "`nSearching for Conda..." -ForegroundColor Yellow
$condaPaths = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "$env:CONDA_EXE"
)

$condaExe = $null
foreach ($path in $condaPaths) {
    if ($path -and (Test-Path $path)) {
        $condaExe = $path
        Write-Host "Conda found: $path" -ForegroundColor Green
        break
    }
}

if (-not $condaExe) {
    Write-Host "Conda not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Initialize Conda
$condaBase = Split-Path (Split-Path $condaExe -Parent) -Parent
$hookScript = "$condaBase\shell\condabin\conda-hook.ps1"

if (Test-Path $hookScript) {
    Write-Host "Initializing Conda..." -ForegroundColor Yellow
    . $hookScript
} else {
    Write-Host "Conda hook script not found: $hookScript" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate environment
Write-Host "Activating environment '$EnvName'..." -ForegroundColor Yellow
conda activate $EnvName

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install Claude Code
# Write-Host "`nInstalling Claude Code $Version..." -ForegroundColor Yellow
# npm install -g "@anthropic-ai/claude-code@$Version"

# if ($LASTEXITCODE -ne 0) {
#     Write-Host "Installation failed" -ForegroundColor Red
#     Read-Host "Press Enter to exit"
#     exit 1
# }

# Run Claude
Write-Host "`nStarting Claude..." -ForegroundColor Green
Write-Host "-------------------------------`n" -ForegroundColor Gray

claude --dangerously-skip-permissions

Write-Host "`n-------------------------------" -ForegroundColor Gray
Write-Host "Claude has exited." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
