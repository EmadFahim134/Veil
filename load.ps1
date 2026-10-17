<#
.SYNOPSIS
  Loads the variables from your .env file into the current PowerShell session.

.DESCRIPTION
  Reads KEY=VALUE lines from .env (next to this script) and sets them as
  environment variables, so scripts and tools can use $env:HF_TOKEN.
  It never prints secret values. Variables last until you close the terminal.

.EXAMPLE
  ./load.ps1
  ./get-veil-data.ps1
#>
param(
    [string]$EnvFile = (Join-Path $PSScriptRoot ".env")
)

if (-not (Test-Path $EnvFile)) {
    Write-Warning "No .env file found at $EnvFile"
    Write-Host "Create one:  Copy-Item .env.example .env   then paste your Hugging Face token into it."
    return
}

$loaded = @()
foreach ($line in Get-Content $EnvFile) {
    $line = $line.Trim()
    if (-not $line -or $line.StartsWith("#")) { continue }
    if ($line -notmatch '^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { continue }

    $name  = $Matches[1]
    $value = $Matches[2].Trim()
    if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") { $value = $Matches[1] }

    Set-Item -Path "Env:$name" -Value $value
    $loaded += $name
}

# Hugging Face tools look for HF_TOKEN. Accept API_KEY as an alias.
if (-not $env:HF_TOKEN -and $env:API_KEY) {
    $env:HF_TOKEN = $env:API_KEY
    $loaded += "HF_TOKEN (from API_KEY)"
}

if ($loaded.Count -eq 0) {
    Write-Warning "$EnvFile has no KEY=VALUE lines."
} else {
    Write-Host "Loaded: $($loaded -join ', ')" -ForegroundColor Green
}

# Safety check: the .env file must never be committed.
if (Get-Command git -ErrorAction SilentlyContinue) {
    git check-ignore -q $EnvFile 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ".env is NOT listed in .gitignore. Run:  Add-Content .gitignore '.env'"
    }
}
