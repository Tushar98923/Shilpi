# Starts the local speech-to-text server for the add-on's mic button (Whisper base.en, CPU, ~75 MB).
#
# Usage:   .\scripts\start_asr_server.ps1
#          .\scripts\start_asr_server.ps1 -Model small.en      # more accurate, ~250 MB, slower
#
# First time only:  py -3.12 -m venv .venv-asr; .venv-asr\Scripts\python -m pip install faster-whisper sounddevice
param(
    [string]$Model = "base.en",
    [int]$Port = 8081
)
$python = "$PSScriptRoot\..\.venv-asr\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Missing .venv-asr. Run: py -3.12 -m venv .venv-asr; .venv-asr\Scripts\python -m pip install faster-whisper sounddevice"
    exit 1
}
& $python "$PSScriptRoot\asr_server.py" --model $Model --port $Port
