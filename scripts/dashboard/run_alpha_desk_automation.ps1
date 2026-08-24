$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stockRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appRoot = Join-Path $repoRoot "alpha-desk-cloud"
$monitorRoot = Join-Path $appRoot "cloudflare-monitor"
$wrangler = Join-Path $appRoot "node_modules\.bin\wrangler.CMD"
$nodeBin = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
$statusPath = Join-Path $stockRoot "Results\mobile_investment_app\weekly_publish_status.json"
$logPath = Join-Path $stockRoot "Results\mobile_investment_app\weekly_publish.log"

New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
$env:Path = "$nodeBin;$env:Path"
# Task Scheduler starts Windows PowerShell with the legacy system code page.
# The repository and result paths contain Korean characters, so both Python's
# own diagnostics and child-process output must be UTF-8 or a harmless log
# message can abort the entire weekly publication with UnicodeEncodeError.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

try {
    # Native tools write harmless build warnings to stderr.  PowerShell turns
    # those into ErrorRecord objects when Stop is active, which would kill a
    # healthy publication before its real exit code is available.
    $ErrorActionPreference = "Continue"
    $publishExitCode = 1
    $attemptDelays = @(0, 30, 60)
    for ($attempt = 0; $attempt -lt $attemptDelays.Count; $attempt++) {
        if ($attemptDelays[$attempt] -gt 0) {
            "[$((Get-Date).ToString('o'))] retry $($attempt + 1)/$($attemptDelays.Count) after $($attemptDelays[$attempt]) seconds" | Add-Content -LiteralPath $logPath
            Start-Sleep -Seconds $attemptDelays[$attempt]
        }
        & $python (Join-Path $repoRoot "scripts\dashboard\publish_alpha_desk.py") --stock-root $stockRoot *>> $logPath
        $publishExitCode = $LASTEXITCODE
        if ($publishExitCode -eq 0) { break }
    }
    $ErrorActionPreference = "Stop"
    if ($publishExitCode -ne 0) { throw "Alpha Desk publication exited with code $publishExitCode" }
}
catch {
    # Always replace transient "running" state. Leaving it behind makes the
    # dashboard continue showing the previous successful Friday as healthy.
    $failure = @{
        status = "error"
        checkedAt = (Get-Date).ToUniversalTime().ToString("o")
        error = $_.Exception.Message
    } | ConvertTo-Json
    $temporary = "$statusPath.tmp"
    [System.IO.File]::WriteAllText($temporary, $failure, [System.Text.UTF8Encoding]::new($false))
    Move-Item -Force -LiteralPath $temporary -Destination $statusPath
    throw
}
finally {
    if (Test-Path -LiteralPath $statusPath) {
        & $wrangler kv key put weekly-publish-status --binding MONITOR_STATUS --remote --path $statusPath --config (Join-Path $monitorRoot "wrangler.jsonc") *>> $logPath
        if ($LASTEXITCODE -ne 0) { throw "Weekly status upload exited with code $LASTEXITCODE" }
    }
}
