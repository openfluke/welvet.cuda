@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if defined WELVET_GO_ROOT (
  set "WELVET_GO=%WELVET_GO_ROOT%"
) else (
  set "WELVET_GO=%ROOT%\..\welvet"
)
for %%I in ("%WELVET_GO%") do set "WELVET_GO=%%~fI"

if not exist "%WELVET_GO%\go.mod" (
  echo Go Welvet checkout not found: %WELVET_GO% >&2
  exit /b 1
)

call "%~dp0build_windows.cmd"
if errorlevel 1 exit /b 1

if not defined BUILD_DIR set "BUILD_DIR=%ROOT%\build-windows"
set "DLL=%BUILD_DIR%\libwelvet_cuda.dll"
if not exist "%DLL%" set "DLL=%BUILD_DIR%\Release\libwelvet_cuda.dll"
if not exist "%DLL%" (
  echo libwelvet_cuda.dll not found under %BUILD_DIR% >&2
  exit /b 1
)
for %%I in ("%DLL%") do set "OUTDIR=%%~dpI"
set "TOOL=%OUTDIR%welvet_dense_fwd.exe"
if not exist "%TOOL%" (
  echo welvet_dense_fwd.exe not found beside DLL at %OUTDIR% >&2
  exit /b 1
)

set "DEST=%WELVET_GO%\native\cuda\windows_amd64"
mkdir "%DEST%" >nul 2>&1
mkdir "%WELVET_GO%\native\cuda\include" >nul 2>&1

copy /Y "%DLL%" "%DEST%\libwelvet_cuda.dll" >nul
if exist "%OUTDIR%libwelvet_cuda.lib" copy /Y "%OUTDIR%libwelvet_cuda.lib" "%DEST%\libwelvet_cuda.lib" >nul
if exist "%OUTDIR%welvet_cuda.lib" copy /Y "%OUTDIR%welvet_cuda.lib" "%DEST%\welvet_cuda.lib" >nul
copy /Y "%ROOT%\include\welvet_c.h" "%WELVET_GO%\native\cuda\include\welvet_cuda.h" >nul
copy /Y "%ROOT%\include\welvet_c.h" "%DEST%\welvet_c.h" >nul

for /f "usebackq delims=" %%V in (`"%TOOL%" --version`) do set "VERSION=%%V"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0write_windows_manifest.ps1" -WelvetGo "%WELVET_GO%" -Root "%ROOT%" -Dest "%DEST%" -Version "%VERSION%"
if errorlevel 1 exit /b 1

echo Installed %DEST%\libwelvet_cuda.dll
echo Go build: go build -tags welvet_cuda ./...
endlocal
