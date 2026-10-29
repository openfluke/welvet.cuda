# Thin PowerShell wrapper around windows_install_go.cmd.
$ErrorActionPreference = "Stop"
$cmd = Join-Path $PSScriptRoot "windows_install_go.cmd"
& cmd /c "`"$cmd`""
exit $LASTEXITCODE
