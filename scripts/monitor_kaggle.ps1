param([string]$Workspace = "C:\Users\kkaso\Documents\Scripts\Sol")
Set-Location -LiteralPath $Workspace
& (Join-Path $Workspace '.venv\Scripts\python.exe') (Join-Path $Workspace 'scripts\monitor_training.py')