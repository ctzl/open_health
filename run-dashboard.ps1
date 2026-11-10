# run-dashboard.ps1 - start the local Oura health dashboard.
# Serves http://127.0.0.1:8090 using the data synced by the sibling
# open_oura checkout. All data stays on this machine.
#
# Usage:
#   .\run-dashboard.ps1              # normal start (localhost only)
#   .\run-dashboard.ps1 -Lan         # also reachable from other LAN devices
#   .\run-dashboard.ps1 -Build       # rebuild the binary first
#   .\run-dashboard.ps1 -Port 8095   # alternate port
#
# Blood lab PDFs (per-profile dirs <serial>\*.pdf + blood.db) are served from
# ~\Documents\official\health by default; override with -BloodDir.

param(
    [switch]$Build,
    [switch]$Lan,
    [int]$Port = 8090,
    [string]$BloodDir = (Join-Path $HOME 'Documents\official\health')
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
    Write-Error "oura.exe not built yet. Run: .\un-dashboard.ps1 -Build"
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
$bloodArgs = @()
if ($BloodDir -and (Test-Path $BloodDir)) {
    $bloodArgs = @('--blood-files', $BloodDir)
    $pdfs = @(Get-ChildItem -Path $BloodDir -Filter '*.pdf' -Recurse -ErrorAction SilentlyContinue)
    Write-Host "blood: $($pdfs.Count) report(s) from $BloodDir"
} elseif ($BloodDir) {
    Write-Warning "blood dir not found: $BloodDir (blood panel disabled)"
}
$lanArgs = @()
if ($Lan) {
    $lanArgs = @('--lan')
    $ips = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' }).IPAddress
    Write-Host "LAN: open http://<$($ips -join '> or http://<')>:$Port from other devices"
    Write-Host "note: Windows may prompt for a firewall rule on first LAN connection"
}
Write-Host "open http://127.0.0.1:$Port  (Ctrl+C to stop)"
# Keep the operational log in a file, not just in the terminal session
# (startup lines, blood backfill/imports, model recomputes, errors).
# Tee-Object writes UTF-16 and keeps ANSI colors, which corrupts the existing
# UTF-8 log — append UTF-8 lines with the color codes stripped instead.
$log = Join-Path (Split-Path $repo -Parent) 'open_oura\dashboard-server.log'
Write-Host "log: $log"
# The binary emits UTF-8; decode its output as UTF-8 (not the OEM codepage)
# so em-dashes and timestamps survive the tee.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$esc = [char]27
& $exe dashboard --db $db --key-file $keyFile --tz-offset $tzOffset --port $Port @bloodArgs @lanArgs 2>&1 |
    ForEach-Object {
        $line = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { [string]$_ }
        Write-Output $line
        Add-Content -Path $log -Value ($line -replace "$esc\[[0-9;]*m", '') -Encoding utf8
    }
