# Thin PowerShell wrapper around build_windows.cmd (VsDevCmd + Ninja + nvcc).
$ErrorActionPreference = "Stop"
$cmd = Join-Path $PSScriptRoot "build_windows.cmd"
& cmd /c "`"$cmd`""
exit $LASTEXITCODE
