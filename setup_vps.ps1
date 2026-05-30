# ============================================================================
#  One-shot VPS setup for the XAU signal bot (Windows Server / Windows VPS).
#  Installs Python, downloads the repo, installs deps, and writes the Telegram
#  config. MT5 (Exness) must be installed + logged in MANUALLY (broker login
#  can't be scripted). Run in an ADMIN PowerShell:
#
#      Set-ExecutionPolicy -Scope Process Bypass -Force
#      iwr -useb https://raw.githubusercontent.com/rodspep/Copy/main/setup_vps.ps1 | iex
#
#  ...or download this file and:  powershell -ExecutionPolicy Bypass -File setup_vps.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$InstallDir = "C:\mt5-bot"
$PyVer      = "3.12.7"
$RepoZip    = "https://github.com/rodspep/Copy/archive/refs/heads/main.zip"

function Info($m){ Write-Host "`n==> $m" -ForegroundColor Cyan }

# --- 1. Python -------------------------------------------------------------
Info "Checking Python..."
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) {
    Info "Installing Python $PyVer (silent)..."
    $exe = "$env:TEMP\python-$PyVer-amd64.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe" -OutFile $exe
    Start-Process $exe -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}
# resolve python path robustly
$PYEXE = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PYEXE) { $PYEXE = "C:\Program Files\Python312\python.exe" }
if (-not (Test-Path $PYEXE)) { throw "Python not found after install" }
& $PYEXE --version

# --- 2. Repo ---------------------------------------------------------------
Info "Downloading repo to $InstallDir ..."
$zip = "$env:TEMP\copy-main.zip"
Invoke-WebRequest $RepoZip -OutFile $zip
if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
Expand-Archive $zip -DestinationPath "$env:TEMP\copy-extract" -Force
Move-Item "$env:TEMP\copy-extract\Copy-main" $InstallDir -Force
Remove-Item $zip,"$env:TEMP\copy-extract" -Recurse -Force -ErrorAction SilentlyContinue
Set-Location $InstallDir

# --- 3. Python deps --------------------------------------------------------
Info "Installing Python packages (this takes 1-2 min)..."
& $PYEXE -m pip install --upgrade pip --quiet
& $PYEXE -m pip install -r requirements.txt MetaTrader5 --quiet
Info "Packages installed."

# --- 4. Telegram config (secrets entered here, NOT stored in repo) ---------
Info "Telegram config"
$cfgPath = "$InstallDir\configs\telegram.json"
if (Test-Path $cfgPath) {
    Write-Host "configs\telegram.json already exists — leaving it as is."
} else {
    $token = Read-Host "Paste your TELEGRAM BOT TOKEN"
    $chat  = Read-Host "Paste your TELEGRAM CHAT ID (e.g. -1003968679027)"
    $cfg = @{ bot_token=$token; chat_id=$chat; timeframes=@("H1","M30");
              risk_pct=1.0; price_offset=0.0 }
    ($cfg | ConvertTo-Json) | Set-Content -Encoding UTF8 $cfgPath
    Write-Host "Wrote $cfgPath"
}

# --- 5. Done ---------------------------------------------------------------
Write-Host "`n=====================================================" -ForegroundColor Green
Write-Host " SETUP DONE. Remaining MANUAL steps:" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host @"
 1. Install the Exness MetaTrader 5 terminal and LOG IN.
    Add your gold symbol (e.g. XAUUSDm) to Market Watch.
 2. If your gold symbol is NOT 'XAUUSDm', edit run_bot_vps.bat:
       set MT5_SYMBOL=YOUR_SYMBOL
 3. Test the data feed:
       cd $InstallDir
       python -X utf8 scripts\check_mt5.py
 4. Start the bot:
       run_bot_vps.bat
 5. Auto-start on reboot: Task Scheduler -> At log on -> run_bot_vps.bat
"@ -ForegroundColor White
