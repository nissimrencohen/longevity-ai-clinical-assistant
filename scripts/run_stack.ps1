<#
.SYNOPSIS
    Boot the three host services for the Longevity clinical assistant.

.DESCRIPTION
    Starts, in the background and without blocking the terminal:
      * MLflow risk-router model server   127.0.0.1:5001
      * FastAPI clinical backend          127.0.0.1:8001
      * FastMCP server                    0.0.0.0:<MCP_PORT from .env>

    Three properties this script is careful about, each learned the hard way:

    1. TRULY DETACHED. Services are launched via WMI (Win32_Process.Create), not
       Start-Process. A Start-Process child is created inside the launching
       shell's job object, so it is killed when that shell exits - which silently
       took down the whole stack once. A WMI-created process has no such parent
       relationship and survives.

    2. NEVER KILLS A PROCESS IT DOES NOT OWN. Freeing a port by killing whatever
       is listening is dangerous: port 9000 was once taken by a QuestDB container
       from an unrelated project, and blindly killing the listener would have
       taken out com.docker.backend. This script only stops listeners whose
       process name is in $OwnedProcessNames, and otherwise aborts with an
       explanation.

    3. HEALTH-CHECKS, NOT PORT-CHECKS. "Something is listening on the port" is
       not the same as "our service is up" - that exact false positive masked the
       QuestDB collision. Readiness requires the expected HTTP status.

    Logs go to logs/<service>.log and logs/<service>.err.log.

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

# Only these may be stopped to free a port. Anything else is someone else's.
$OwnedProcessNames = @('python', 'pythonw', 'uv', 'uvicorn', 'mlflow')

function Get-DotEnvValue {
    param([string]$Key, [string]$Default)
    $envFile = Join-Path $RepoRoot '.env'
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match "^\s*$Key\s*=\s*(.+?)\s*$") { return $matches[1] }
        }
    }
    return $Default
}

# Read the MCP port from .env so this script and the server cannot disagree.
$McpPort = [int](Get-DotEnvValue -Key 'MCP_PORT' -Default '9100')

$Services = @(
    @{ Name = 'mlflow'; Port = 5001
       Health = 'http://127.0.0.1:5001/ping'; Expect = @(200)
       Cmd = 'uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --host 127.0.0.1 --env-manager local' }
    @{ Name = 'backend'; Port = 8001
       Health = 'http://127.0.0.1:8001/health'; Expect = @(200)
       Cmd = 'uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001' }
    # 401 is the healthy answer for the MCP server: it proves an authenticating
    # FastMCP is listening. An impostor on the port (QuestDB answered 404) fails.
    @{ Name = 'mcp'; Port = $McpPort
       Health = "http://127.0.0.1:$McpPort/mcp"; Expect = @(401)
       Cmd = 'uv run python mcp-server/server.py' }
)

function Get-HttpStatus {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 5 -UseBasicParsing
        return [int]$r.StatusCode
    } catch [System.Net.WebException] {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return 0
    } catch {
        return 0
    }
}

function Get-PortOwner {
    param([int]$Port)
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
         Select-Object -First 1
    if (-not $c) { return $null }
    return Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
}

function Stop-OurPort {
    param([int]$Port, [string]$Name)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        if (-not $p) { continue }
        if ($OwnedProcessNames -contains $p.ProcessName) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Write-Host ("  freed :{0} (was {1} pid {2})" -f $Port, $p.ProcessName, $p.Id)
        } else {
            throw ("Port {0} (for '{1}') is held by '{2}' (pid {3}), which this script does not own. " -f
                   $Port, $Name, $p.ProcessName, $p.Id) +
                  "Refusing to kill it. Stop that process yourself, or change the port in .env."
        }
    }
}

if ($Status) {
    Write-Host "Service status:"
    foreach ($s in $Services) {
        $code = Get-HttpStatus $s.Health
        $owner = Get-PortOwner $s.Port
        if ($s.Expect -contains $code) {
            $state = "UP"
        } elseif ($owner) {
            $state = "WRONG SERVICE on port (owner: $($owner.ProcessName), http $code)"
        } else {
            $state = "down"
        }
        Write-Host ("  {0,-8} :{1,-6} {2}" -f $s.Name, $s.Port, $state)
    }
    exit 0
}

if ($Stop) {
    Write-Host "Stopping host services..."
    foreach ($s in $Services) { Stop-OurPort -Port $s.Port -Name $s.Name }
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
    Stop-OurPort -Port $s.Port -Name $s.Name

    $out = Join-Path $LogDir ("{0}.log" -f $s.Name)
    $err = Join-Path $LogDir ("{0}.err.log" -f $s.Name)

    # WMI-created so the process is not in this shell's job object (see notes).
    $commandLine = 'cmd.exe /c cd /d "{0}" && {1} > "{2}" 2> "{3}"' -f $RepoRoot, $s.Cmd, $out, $err
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $commandLine }

    if ($result.ReturnValue -ne 0) {
        Write-Host ("  {0,-8} FAILED to launch (WMI code {1})" -f $s.Name, $result.ReturnValue)
        continue
    }
    Set-Content -Path (Join-Path $LogDir ("{0}.pid" -f $s.Name)) -Value $result.ProcessId -Encoding ascii
    Write-Host ("  {0,-8} launched -> :{1}" -f $s.Name, $s.Port)
}

Write-Host ""
Write-Host "Waiting for services to pass their health checks..."
$deadline = (Get-Date).AddSeconds(180)
$pending = [System.Collections.ArrayList]@($Services)

while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    foreach ($s in @($pending)) {
        $code = Get-HttpStatus $s.Health
        if ($s.Expect -contains $code) {
            Write-Host ("  {0,-8} healthy on :{1} (http {2})" -f $s.Name, $s.Port, $code)
            $pending.Remove($s)
        }
    }
}

if ($pending.Count -gt 0) {
    Write-Host ""
    Write-Host "TIMED OUT waiting for:" -ForegroundColor Red
    foreach ($s in $pending) {
        Write-Host ("  {0} (:{1}) - see {2}\{0}.err.log" -f $s.Name, $s.Port, $LogDir)
    }
    exit 1
}

Write-Host ""
Write-Host "All three host services are up." -ForegroundColor Green
foreach ($s in $Services) { Write-Host ("  {0,-8} :{1}" -f $s.Name, $s.Port) }
Write-Host ("Logs: {0}" -f $LogDir)
