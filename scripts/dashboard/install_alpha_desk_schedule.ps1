$ErrorActionPreference = "Stop"

# Name kept as-is so Register-ScheduledTask -Force replaces the existing
# registration rather than leaving a second task running the same publish.
$taskName = "AlphaDesk Weekly Publish"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "scripts\dashboard\run_alpha_desk_automation.ps1"
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
# Selection is weekly, but marks are daily.  The Friday/Saturday pair still
# covers the weekly signal cut; the weekday trigger keeps the bundled price
# history on the latest close so mid-week sessions are valued.  6:00 PM local
# is well past the 1:00 PM PT bell, by which point the vendor has settled the
# closing print.  Same-day repeats are cheap: publish_alpha_desk.py stops
# before the export when neither the signal nor the last close advanced.
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "6:00 PM"),
    (New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Friday -At "10:55 PM"),
    (New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Saturday -At "7:00 AM"),
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)
)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $triggers `
    -Settings $settings `
    -Principal $principal `
    -Description "Alpha Desk weekly signal, daily price refresh, ticker review, data publication, and automatic recovery" `
    -Force | Out-Null

Get-ScheduledTask -TaskName $taskName
