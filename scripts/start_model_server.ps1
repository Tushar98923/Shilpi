# Starts the local model server that the Blender add-on and eval script talk to.
# Leave this window open while using the add-on; close it to stop the server.
#
#   Right-click this file > "Run with PowerShell"
#   or:  .\scripts\start_model_server.ps1                 (the v5 operator model)
#        .\scripts\start_model_server.ps1 -Model models\operator\qwen3.5-2b-operator-v4-q8_0.gguf

param(
    [string]$Model = "models\operator\qwen3.5-2b-operator-v5-q8_0.gguf",
    [int]$Port = 8080
)

# A relative path works from any folder: it is looked up from the current folder, then from the project folder.
if (-not (Test-Path $Model)) { $Model = Join-Path "$PSScriptRoot\.." $Model }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

$exe = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\llama-server.exe"
if (-not (Test-Path $exe)) { $exe = (Get-Command llama-server -ErrorAction SilentlyContinue).Source }
if (-not $exe) {
    # Right after a winget install, PATH isn't refreshed yet; look in winget's package folder.
    $exe = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1).FullName
}
if (-not $exe) { throw "llama-server not found. Install it with: winget install ggml.llamacpp" }

# Passed via env var: PowerShell mangles the quotes when JSON goes on the command line.
# Thinking must be off -- the model was trained to answer with JSON immediately.
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"enable_thinking": false}'

Write-Host "Starting model server on http://127.0.0.1:$Port  (Ctrl+C or close this window to stop)"
& $exe -m $Model --port $Port --jinja -ngl 99 -c 2048
