@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%ROOT%\build-windows"

if not defined CUDAToolkit_ROOT (
  if defined CUDA_PATH set "CUDAToolkit_ROOT=%CUDA_PATH%"
)
if not defined CUDAToolkit_ROOT (
  for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDAToolkit_ROOT=%%~fD"
)
if not defined CUDAToolkit_ROOT (
  echo nvcc not found. Install the NVIDIA CUDA Toolkit, or set CUDAToolkit_ROOT / CUDA_PATH. >&2
  exit /b 1
)
if not exist "%CUDAToolkit_ROOT%\bin\nvcc.exe" (
  echo nvcc not found under "%CUDAToolkit_ROOT%". >&2
  exit /b 1
)
set "CUDA_PATH=%CUDAToolkit_ROOT%"
set "CudaToolkitDir=%CUDAToolkit_ROOT%\"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%I"
if not defined VS (
  echo Visual Studio C++ toolset not found. >&2
  exit /b 1
)

set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "VSDEVCMD=%VS%\Common7\Tools\VsDevCmd.bat"
if not defined CMAKE_CUDA_ARCHITECTURES set "CMAKE_CUDA_ARCHITECTURES=75;86"

echo CUDAToolkit_ROOT=%CUDAToolkit_ROOT%
echo host: MSVC via %VS%
echo CUDA archs: %CMAKE_CUDA_ARCHITECTURES%
echo build dir: %BUILD_DIR%

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 (
  echo VsDevCmd failed. >&2
  exit /b 1
)

set "PATH=%CUDAToolkit_ROOT%\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"

"%CMAKE%" -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_CUDA_COMPILER="%CUDAToolkit_ROOT%\bin\nvcc.exe" -DCMAKE_CUDA_HOST_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CUDA_ARCHITECTURES=%CMAKE_CUDA_ARCHITECTURES% -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%"
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%BUILD_DIR%" -j
if errorlevel 1 exit /b 1

echo built: %BUILD_DIR%
echo   lib:  %BUILD_DIR%\libwelvet_cuda.dll
echo   tool: %BUILD_DIR%\welvet_dense_fwd.exe
echo   ex:   %BUILD_DIR%\example_cuda_smoke.exe
echo.
echo smoke:  %BUILD_DIR%\example_cuda_smoke.exe
echo gaps:   %BUILD_DIR%\welvet_dense_fwd.exe --gaps
endlocal
