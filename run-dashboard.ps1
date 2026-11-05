# run-dashboard.ps1 - start the local Oura health dashboard.
# Serves http://127.0.0.1:8090 using the data synced by the sibling
# open_oura checkout. All data stays on this machine.
#
# Usage:
#   .\run-dashboard.ps1                 # normal start
#   .\run-dashboard.ps1 -Build          # rebuild the binary first
#   .\run-dashboard.ps1 -Port 8095      # alternate port

param(
    [switch]$Build,
    [int]$Port = 8090
)

$ErrorActionPreference = 'Stop'

$repo    = $PSScriptRoot
$db      = Join-Path (Split-Path $repo -Parent) 'open_oura\oura.db'
$keyFile = Join-Path (Split-Path $repo -Parent) 'open_oura\key.hex'
$exe     = Join-Path $repo 'target\release\oura.exe'

# --- sanity checks -----------------------------------------------------------
if (-not (Test-Path $db))      { Write-Error "database not found: $db";      exit 1 }
if (-not (Test-Path $keyFile)) { Write-Error "auth key not found: $keyFile"; exit 1 }

if ($Build) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        $env:PATH = "$HOME\.cargo\bin;$env:PATH"   # rustup default install location
    }
    Write-Host 'building oura-cli (release)...'
    Push-Location $repo
    try { cargo build --release -p oura-cli; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
    finally { Pop-Location }
}

if (-not (Test-Path $exe)) {
    Write-Error "oura.exe not built yet. Run: .\run-dashboard.ps1 -Build"
    exit 1
}

# warn if something is already listening on the port (usually a previous dashboard)
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Warning "port $Port is already in use - a dashboard may already be running at http://127.0.0.1:$Port"
    exit 1
}

# the CLI defaults to UTC on Windows, so pass the real local offset (DST-aware)
$tzOffset = [TimeZoneInfo]::Local.GetUtcOffset([DateTimeOffset]::Now).TotalHours

Write-Host "tz offset: $tzOffset  db: $db"
Write-Host "open http://127.0.0.1:$Port  (Ctrl+C to stop)"
& $exe dashboard --db $db --key-file $keyFile --tz-offset $tzOffset --port $Port
