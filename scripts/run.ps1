<#
.SYNOPSIS
    Experiment run wrapper: tee stdout+stderr to logs\<timestamp>_<name>.log
    and register run status in logs\runs.jsonl for tools/dashboard.

.DESCRIPTION
    Constraints measured on this machine:
      - Sandbox runs Windows PowerShell 5.1 + ConstrainedLanguage: `pwsh` is not
        on PATH and [scriptblock]::Create / Invoke-Expression are blocked, so we
        never nest a shell.
      - This file is intentionally ASCII-only: Windows PowerShell 5.1 decodes
        BOM-less UTF-8 as ANSI, and non-ASCII characters break parsing.
      - Experiment scripts are NOT modified; this wrapper only redirects output.

.EXAMPLE
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1 `
        -Name train_E1_seed13 `
        -Exe "C:\path\python.exe" `
        -Args "src\train\train_arm.py","--arm","E1","--seed","13"
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Exe,
    [string[]]$Args = @(),
    [string]$WorkDir = ""
)

$ws = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $ws "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$id = "${stamp}_$Name"
$file = "$id.log"
$logPath = Join-Path $logDir $file
$runsPath = Join-Path $logDir "runs.jsonl"
$start = (Get-Date).ToString("o")
$cmdText = "$Exe $($Args -join ' ')"

$entry = [ordered]@{
    id = $id; name = $Name; file = $file; cmd = $cmdText
    start = $start; end = $null; exit = $null; status = "running"
}
($entry | ConvertTo-Json -Compress) | Add-Content -Path $runsPath -Encoding utf8

@(
    "=== $Name ==="
    "id      : $id"
    "start   : $start"
    "workdir : $ws"
    "command : $cmdText"
    "----------------------------------------"
) | Out-File -FilePath $logPath -Encoding utf8

if ($WorkDir -ne "") { Set-Location $WorkDir } else { Set-Location $ws }
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[run] $Name"
Write-Host "[run] log : $logPath"
Write-Host "[run] dash: http://127.0.0.1:8765/"

# Merge every stream into stdout, then tee live to the file.
& $Exe @Args *>&1 | Tee-Object -FilePath $logPath -Append | ForEach-Object { $_ }
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 0 }

$end = (Get-Date).ToString("o")
$status = if ($code -eq 0) { "done" } else { "failed" }

@(
    "----------------------------------------"
    "end     : $end"
    "exit    : $code"
    "status  : $status"
) | Out-File -FilePath $logPath -Append -Encoding utf8

$done = [ordered]@{
    id = $id; name = $Name; file = $file; cmd = $cmdText
    start = $start; end = $end; exit = $code; status = $status
}
($done | ConvertTo-Json -Compress) | Add-Content -Path $runsPath -Encoding utf8

Write-Host "[run] exit=$code status=$status"
exit $code
