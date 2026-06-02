# Reliable start/stop of the UG feeds (listener + copier) on the MT5 VPS.
# - Scopes strictly to python.exe in the MT5 interactive session whose command line
#   is `-m scripts.ug_reader|ug_copier` AND under C:\mt5-bot (never touches the bot
#   or the session-0 volscan bot).
# - taskkill /F /T (tree) in a loop until ZERO remain, then fails hard.
# - start-feeds verifies zero before launching, then sanity-checks the count.
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("stop-feeds", "start-feeds")]
  [string]$Action
)
$ErrorActionPreference = "Stop"
$BotDir = "C:\mt5-bot"

function Get-Mt5SessionId {
  $s = @(Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq "terminal64.exe" -and $_.SessionId -ne 0 } |
    Select-Object -ExpandProperty SessionId -Unique)
  if ($s.Count -ne 1) { throw "Need exactly one interactive MT5 terminal; found $($s.Count)." }
  return [int]$s[0]
}

function Get-FeedProcesses {
  $sid = Get-Mt5SessionId
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and $_.SessionId -eq $sid -and $_.CommandLine -and
    $_.CommandLine -like "*$BotDir*" -and
    ($_.CommandLine -match '(^| )-m scripts\.ug_reader(\s|$)' -or
     $_.CommandLine -match '(^| )-m scripts\.ug_copier(\s|$)')
  }
}

function Stop-FeedProcesses {
  for ($i = 0; $i -lt 20; $i++) {
    $procs = @(Get-FeedProcesses)
    if ($procs.Count -eq 0) { return }
    foreach ($p in $procs) { & taskkill.exe /F /T /PID $p.ProcessId 2>$null | Out-Null }
    Start-Sleep -Milliseconds 750
  }
  $left = @(Get-FeedProcesses)
  if ($left.Count -ne 0) {
    $left | Select-Object ProcessId, CommandLine | Format-List | Out-String | Write-Host
    throw "Feed processes still running after kill loop."
  }
}

Stop-FeedProcesses
Remove-Item -Force "$BotDir\data\ug\copier_live.lock", "$BotDir\data\ug\copier_dry.lock" -ErrorAction SilentlyContinue
if ($Action -eq "stop-feeds") { Write-Host "feeds stopped (0 remaining)."; exit 0 }

# start-feeds: launch one listener + one copier, then sanity-check the count.
& "$BotDir\run_listener.bat"
& "$BotDir\run_copier.bat"
Start-Sleep -Seconds 6
$n = @(Get-FeedProcesses).Count
Write-Host "feed python processes after launch: $n"
if ($n -lt 2 -or $n -gt 6) { throw "Unexpected feed process count: $n (expected ~4 = listener+copier x stub+real)" }
