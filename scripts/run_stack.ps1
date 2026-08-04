<#
.SYNOPSIS
    Boot the three host services for the Longevity clinical assistant.

.DESCRIPTION
    Starts, in the background and without blocking the terminal:
      * MLflow risk-router model server   127.0.0.1:5001
      * FastAPI clinical backend          127.0.0.1:8001
      * FastMCP server                    0.0.0.0:9000

    Idempotent: frees each port before binding it, so re-running restarts the
    stack cleanly rather than colliding with an old process.

    Logs go to logs/<service>.log; PIDs to logs/<service>.pid.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1
    powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1 -Stop
    powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1 -Status
#>
param(
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Services = @(
    @{ Name = 'mlflow'; Port = 5001; Url = 'http://127.0.0.1:5001/ping'
       Args = @('run', 'mlflow', 'models', 'serve', '-m', 'models/mlflow_risk_router',
                '-p', '5001', '--host', '127.0.0.1', '--env-manager', 'local') }
    @{ Name = 'backend'; Port = 8001; Url = 'http://127.0.0.1:8001/health'
       Args = @('run', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8001') }
    @{ Name = 'mcp'; Port = 9000; Url = 'http://127.0.0.1:9000/mcp/'
       Args = @('run', 'python', 'mcp-server/server.py') }
)

function Stop-Port {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop
            Write-Host ("  freed port {0} (pid {1})" -f $Port, $c.OwningProcess)
        } catch {
            Write-Host ("  could not stop pid {0} on port {1}" -f $c.OwningProcess, $Port)
        }
    }
}

function Test-Port {
    param([int]$Port)
    $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if ($Status) {
    Write-Host "Service status:"
    foreach ($s in $Services) {
        $state = if (Test-Port $s.Port) { 'LISTENING' } else { 'down' }
        Write-Host ("  {0,-8} :{1,-5} {2}" -f $s.Name, $s.Port, $state)
    }
    exit 0
}

if ($Stop) {
    Write-Host "Stopping host services..."
    foreach ($s in $Services) { Stop-Port $s.Port }
    Write-Host "Stopped."
    exit 0
}

# The MLflow router is generated from the committed pickles and is gitignored,
# so register it if this is a fresh checkout.
if (-not (Test-Path (Join-Path $RepoRoot 'models/mlflow_risk_router/MLmodel'))) {
    Write-Host "Registering the MLflow risk router (first run)..."
    Push-Location $RepoRoot
    & uv run python models/register_router.py | Out-Null
    Pop-Location
}

Write-Host "Starting host services..."
foreach ($s in $Services) {
    Stop-Port $s.Port

    $out = Join-Path $LogDir ("{0}.log" -f $s.Name)
    $err = Join-Path $LogDir ("{0}.err.log" -f $s.Name)

    $p = Start-Process -FilePath 'uv' -ArgumentList $s.Args `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err

    Set-Content -Path (Join-Path $LogDir ("{0}.pid" -f $s.Name)) -Value $p.Id -Encoding utf8
    Write-Host ("  {0,-8} pid {1} -> :{2}" -f $s.Name, $p.Id, $s.Port)
}

Write-Host "`nWaiting for services to become reachable..."
$deadline = (Get-Date).AddSeconds(180)
$pending = [System.Collections.ArrayList]@($Services)

while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    foreach ($s in @($pending)) {
        if (Test-Port $s.Port) {
            Write-Host ("  {0,-8} ready on :{1}" -f $s.Name, $s.Port)
            $pending.Remove($s)
        }
    }
}

if ($pending.Count -gt 0) {
    Write-Host "`nTIMED OUT waiting for:" -ForegroundColor Red
    foreach ($s in $pending) {
        Write-Host ("  {0} (:{1}) - see {2}\{0}.err.log" -f $s.Name, $s.Port, $LogDir)
    }
    exit 1
}

Write-Host "`nAll three host services are up." -ForegroundColor Green
foreach ($s in $Services) { Write-Host ("  {0,-8} :{1}" -f $s.Name, $s.Port) }
Write-Host ("Logs: {0}" -f $LogDir)
