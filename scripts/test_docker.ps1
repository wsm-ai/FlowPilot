[CmdletBinding()]
param(
    [string]$ComposeFile,
    [string]$ProjectName = ("flowpilot13ftest" + [Guid]::NewGuid().ToString("N").Substring(0, 8)),
    [string]$DockerCommand = "docker",
    [int]$HealthyTimeoutSeconds = 120,
    [string]$PersistenceDatabasePath = "/app/data/stage13f-persistence-test.sqlite"
)

$ErrorActionPreference = "Stop"

$checks = [ordered]@{
    "Preflight" = "NOT RUN"
    "Compose configuration" = "NOT RUN"
    "Docker image build" = "NOT RUN"
    "Container startup" = "NOT RUN"
    "Docker healthy" = "NOT RUN"
    "/health endpoint" = "NOT RUN"
    "API authentication" = "NOT RUN"
    "Non-root user" = "NOT RUN"
    "SQLite writable" = "NOT RUN"
    "Logging configuration" = "NOT RUN"
    "SQLite cross-container persistence" = "NOT RUN"
    "Container cleanup" = "NOT RUN"
}

$currentCheck = $null
$failureMessage = $null
$ownsProject = $false
$safeEnvFile = $null
$exitCode = 1
$environmentNames = @(
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "MCP_SERVERS",
    "FLOWPILOT_AUTH_ENABLED",
    "FLOWPILOT_API_KEY"
)
$originalEnvironment = @{}

function Start-Check {
    param([string]$Name)
    $script:currentCheck = $Name
}

function Complete-Check {
    param([string]$Name)
    $script:checks[$Name] = "PASS"
    Write-Host "[PASS] $Name"
    $script:currentCheck = $null
}

function Invoke-Docker {
    param([string[]]$Arguments)

    & $script:DockerCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed"
    }
}

function Invoke-DockerCapture {
    param([string[]]$Arguments)

    $output = & $script:DockerCommand @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed"
    }
    return (($output | Out-String).Trim())
}

function Get-ComposeArguments {
    param([string[]]$Arguments)

    return @(
        "compose",
        "--env-file", $script:safeEnvFile,
        "-f", $script:ComposeFile,
        "-p", $script:ProjectName
    ) + $Arguments
}

function Get-ServiceContainerId {
    $containerId = Invoke-DockerCapture -Arguments (
        Get-ComposeArguments -Arguments @("ps", "-q", "flowpilot-api")
    )
    if ([string]::IsNullOrWhiteSpace($containerId)) {
        throw "FlowPilot test container was not found"
    }
    return $containerId.Trim()
}

function Wait-ForHealthyContainer {
    param([string]$ContainerId)

    $deadline = [DateTime]::UtcNow.AddSeconds($script:HealthyTimeoutSeconds)
    do {
        $state = Invoke-DockerCapture -Arguments @(
            "inspect",
            "--format", "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}",
            $ContainerId
        )
        $parts = $state.Split("|")
        if ($parts.Count -ne 2 -or $parts[0] -ne "running") {
            throw "FlowPilot test container is not running"
        }
        if ($parts[1] -eq "healthy") {
            return
        }
        if ($parts[1] -eq "unhealthy" -or $parts[1] -eq "missing") {
            throw "FlowPilot test container did not become healthy"
        }
        Start-Sleep -Seconds 2
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "Timed out waiting for a healthy FlowPilot container"
}

function Show-ComposeDiagnostics {
    try {
        $logs = Invoke-DockerCapture -Arguments (
            Get-ComposeArguments -Arguments @("logs", "--tail=80", "flowpilot-api")
        )
        if (-not [string]::IsNullOrWhiteSpace($logs)) {
            Write-Host "--- FlowPilot diagnostic logs ---"
            Write-Host $logs
        }
    }
    catch {
        Write-Warning "Unable to read diagnostic container logs"
    }
}

function Test-PortAvailable {
    param([int]$Port)

    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return -not ($listeners | Where-Object { $_.Port -eq $Port })
}

try {
    Start-Check -Name "Preflight"

    if ([string]::IsNullOrWhiteSpace($ComposeFile)) {
        $repositoryRoot = Split-Path -Parent $PSScriptRoot
        $ComposeFile = Join-Path $repositoryRoot "compose.yaml"
    }
    if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
        throw "compose.yaml was not found"
    }
    $ComposeFile = (Resolve-Path -LiteralPath $ComposeFile).Path

    $dockerfile = Join-Path (Split-Path -Parent $PSScriptRoot) "Dockerfile"
    if (-not (Test-Path -LiteralPath $dockerfile -PathType Leaf)) {
        throw "Dockerfile was not found"
    }
    if (-not (Get-Command $DockerCommand -ErrorAction SilentlyContinue)) {
        throw "Docker command is not available"
    }
    if ($ProjectName -notmatch "^flowpilot13ftest[a-z0-9]+$") {
        throw "Test project name is invalid"
    }
    if ($HealthyTimeoutSeconds -le 0) {
        throw "Healthy timeout must be greater than zero"
    }
    if (-not $PersistenceDatabasePath.StartsWith("/app/data/") -or $PersistenceDatabasePath.Contains("..")) {
        throw "Persistence test path must be contained in /app/data"
    }
    if (-not (Test-PortAvailable -Port 8000)) {
        throw "Host port 8000 is already in use"
    }

    $null = Invoke-DockerCapture -Arguments @("info")
    $null = Invoke-DockerCapture -Arguments @("compose", "version")

    $volumeName = "${ProjectName}_flowpilot_data"
    $existingContainers = Invoke-DockerCapture -Arguments @(
        "ps", "-a", "--quiet", "--filter", "label=com.docker.compose.project=$ProjectName"
    )
    $existingNetworks = Invoke-DockerCapture -Arguments @(
        "network", "ls", "--quiet", "--filter", "label=com.docker.compose.project=$ProjectName"
    )
    $existingVolumes = Invoke-DockerCapture -Arguments @(
        "volume", "ls", "--quiet", "--filter", "name=^${volumeName}$"
    )
    if ($existingContainers -or $existingNetworks -or $existingVolumes) {
        throw "The generated Docker test project already has resources"
    }

    foreach ($name in $environmentNames) {
        $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "ci-test-key", "Process")
    [Environment]::SetEnvironmentVariable("DEEPSEEK_BASE_URL", "https://example.invalid", "Process")
    [Environment]::SetEnvironmentVariable("DEEPSEEK_MODEL", "deepseek-v4-flash", "Process")
    [Environment]::SetEnvironmentVariable("MCP_SERVERS", "[]", "Process")
    [Environment]::SetEnvironmentVariable("FLOWPILOT_AUTH_ENABLED", "true", "Process")
    [Environment]::SetEnvironmentVariable("FLOWPILOT_API_KEY", "flowpilot-docker-test-key", "Process")

    $safeEnvFile = Join-Path ([IO.Path]::GetTempPath()) (
        "flowpilot-13f-" + [Guid]::NewGuid().ToString("N") + ".env"
    )
    $null = New-Item -ItemType File -Path $safeEnvFile -ErrorAction Stop
    Complete-Check -Name "Preflight"

    Start-Check -Name "Compose configuration"
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("config", "--quiet"))
    Complete-Check -Name "Compose configuration"

    Start-Check -Name "Docker image build"
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("build"))
    Complete-Check -Name "Docker image build"

    Start-Check -Name "Container startup"
    $ownsProject = $true
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("up", "-d"))
    $containerId = Get-ServiceContainerId
    Complete-Check -Name "Container startup"

    Start-Check -Name "Docker healthy"
    Wait-ForHealthyContainer -ContainerId $containerId
    Complete-Check -Name "Docker healthy"

    Start-Check -Name "/health endpoint"
    $healthResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 10
    if ($healthResponse.status -ne "ok" -or $healthResponse.service -ne "FlowPilot") {
        throw "FlowPilot health response is invalid"
    }
    Complete-Check -Name "/health endpoint"

    Start-Check -Name "API authentication"
    try {
        $null = Invoke-WebRequest `
            -Uri "http://127.0.0.1:8000/api/v1/chat" `
            -Method Post `
            -ContentType "application/json" `
            -Body '{"message":""}' `
            -UseBasicParsing `
            -TimeoutSec 10
        throw "Protected API accepted a request without credentials"
    }
    catch {
        if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 401) {
            throw
        }
    }
    try {
        $null = Invoke-WebRequest `
            -Uri "http://127.0.0.1:8000/api/v1/chat" `
            -Method Post `
            -Headers @{ Authorization = "Bearer flowpilot-docker-test-key" } `
            -ContentType "application/json" `
            -Body '{"message":""}' `
            -UseBasicParsing `
            -TimeoutSec 10
        throw "Authenticated validation request unexpectedly succeeded"
    }
    catch {
        if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 422) {
            throw
        }
    }
    try {
        $null = Invoke-WebRequest `
            -Uri "http://127.0.0.1:8000/mcp" `
            -Method Post `
            -ContentType "application/json" `
            -Body '{}' `
            -UseBasicParsing `
            -TimeoutSec 10
        throw "Mounted MCP application accepted a request without credentials"
    }
    catch {
        if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 401) {
            throw
        }
    }
    Complete-Check -Name "API authentication"

    Start-Check -Name "Non-root user"
    $uid = Invoke-DockerCapture -Arguments (
        Get-ComposeArguments -Arguments @("exec", "-T", "flowpilot-api", "id", "-u")
    )
    if ($uid.Trim() -eq "0" -or [string]::IsNullOrWhiteSpace($uid)) {
        throw "FlowPilot container is running as root"
    }
    Complete-Check -Name "Non-root user"

    $marker = "stage13f-" + [Guid]::NewGuid().ToString("N")
    Start-Check -Name "SQLite writable"
    $writeCode = @"
import sqlite3, sys
path, marker = sys.argv[1], sys.argv[2]
database = sqlite3.connect(path)
mode = database.execute('PRAGMA journal_mode=WAL').fetchone()[0]
database.execute('CREATE TABLE IF NOT EXISTS persistence_markers (marker TEXT PRIMARY KEY, value TEXT NOT NULL)')
database.execute('INSERT INTO persistence_markers(marker, value) VALUES (?, ?)', (marker, 'container-a'))
database.commit()
assert mode.lower() == 'wal'
assert database.execute('SELECT value FROM persistence_markers WHERE marker = ?', (marker,)).fetchone() == ('container-a',)
database.close()
"@
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @(
        "exec", "-T", "flowpilot-api", "python", "-c", $writeCode,
        $PersistenceDatabasePath, $marker
    ))
    Complete-Check -Name "SQLite writable"

    Start-Check -Name "Logging configuration"
    $logConfigJson = Invoke-DockerCapture -Arguments @(
        "inspect", "--format", "{{json .HostConfig.LogConfig}}", $containerId
    )
    $logConfig = $logConfigJson | ConvertFrom-Json
    if (
        $logConfig.Type -ne "json-file" -or
        $logConfig.Config."max-size" -ne "10m" -or
        $logConfig.Config."max-file" -ne "3"
    ) {
        throw "Docker logging configuration is invalid"
    }
    $logs = Invoke-DockerCapture -Arguments (
        Get-ComposeArguments -Arguments @("logs", "--tail=80", "flowpilot-api")
    )
    if ($logs -notmatch "Uvicorn running" -or $logs -notmatch "GET /health") {
        throw "Expected FlowPilot container logs were not available"
    }
    Complete-Check -Name "Logging configuration"

    Start-Check -Name "SQLite cross-container persistence"
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("down"))
    $retainedVolume = Invoke-DockerCapture -Arguments @(
        "volume", "ls", "--quiet", "--filter", "name=^${volumeName}$"
    )
    if ($retainedVolume.Trim() -ne $volumeName) {
        throw "Docker test volume was not retained"
    }

    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("up", "-d", "--force-recreate"))
    $containerId = Get-ServiceContainerId
    Wait-ForHealthyContainer -ContainerId $containerId

    $readCode = @"
import sqlite3, sys
path, marker = sys.argv[1], sys.argv[2]
database = sqlite3.connect(path)
assert database.execute('SELECT value FROM persistence_markers WHERE marker = ?', (marker,)).fetchone() == ('container-a',)
database.execute('INSERT INTO persistence_markers(marker, value) VALUES (?, ?)', (marker + '-b', 'container-b'))
database.commit()
assert database.execute('SELECT value FROM persistence_markers WHERE marker = ?', (marker + '-b',)).fetchone() == ('container-b',)
database.close()
"@
    Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @(
        "exec", "-T", "flowpilot-api", "python", "-c", $readCode,
        $PersistenceDatabasePath, $marker
    ))
    Complete-Check -Name "SQLite cross-container persistence"

    $exitCode = 0
}
catch {
    $failureMessage = $_.Exception.Message
    if ($currentCheck -and $checks[$currentCheck] -ne "PASS") {
        $checks[$currentCheck] = "FAIL"
        Write-Host "[FAIL] $currentCheck"
    }
    if ($ownsProject) {
        Show-ComposeDiagnostics
    }
}
finally {
    if ($ownsProject) {
        try {
            Invoke-Docker -Arguments (Get-ComposeArguments -Arguments @("down"))
            $checks["Container cleanup"] = "PASS"
            Write-Host "[PASS] Container cleanup"
        }
        catch {
            $checks["Container cleanup"] = "FAIL"
            $exitCode = 1
            Write-Host "[FAIL] Container cleanup"
            if (-not $failureMessage) {
                $failureMessage = "Unable to clean up Docker test containers and network"
            }
        }
    }
    else {
        $checks["Container cleanup"] = "PASS"
        Write-Host "[PASS] Container cleanup (no Docker resources created)"
    }

    if ($safeEnvFile -and (Test-Path -LiteralPath $safeEnvFile)) {
        try {
            Remove-Item -LiteralPath $safeEnvFile -Force -ErrorAction Stop
        }
        catch {
            $exitCode = 1
            if (-not $failureMessage) {
                $failureMessage = "Unable to remove the temporary environment file"
            }
        }
    }

    foreach ($name in $environmentNames) {
        if ($originalEnvironment.ContainsKey($name)) {
            [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
        }
    }
}

Write-Host ""
Write-Host "FlowPilot Docker integration test report"
foreach ($entry in $checks.GetEnumerator()) {
    Write-Host ("{0}: {1}" -f $entry.Key, $entry.Value)
}

if ($exitCode -eq 0 -and -not ($checks.Values -contains "FAIL")) {
    Write-Host "Overall result: PASS"
    Write-Host "Retained test volume: ${ProjectName}_flowpilot_data"
    exit 0
}

if ($failureMessage) {
    Write-Host "Failure: $failureMessage"
}
Write-Host "Overall result: FAIL"
exit 1
