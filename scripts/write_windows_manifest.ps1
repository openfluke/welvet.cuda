param(
  [Parameter(Mandatory = $true)][string]$WelvetGo,
  [Parameter(Mandatory = $true)][string]$Root,
  [Parameter(Mandatory = $true)][string]$Dest,
  [Parameter(Mandatory = $true)][string]$Version
)

$ErrorActionPreference = "Stop"
$dll = Join-Path $Dest "libwelvet_cuda.dll"
$sha = (Get-FileHash -Algorithm SHA256 $dll).Hash.ToLower()
$destUnix = ($Dest -replace '\\', '/')
$rootUnix = ($Root -replace '\\', '/')

@(
  "engine=welvet.cuda"
  "version=$Version"
  "model_abi=1"
  "segment_extension=1"
  "platform=windows_amd64"
  "source_checkout=$rootUnix"
  "$sha  $destUnix/libwelvet_cuda.dll"
) | Set-Content -Path (Join-Path $WelvetGo "native\cuda\MANIFEST.txt") -Encoding ascii

Set-Content -Path (Join-Path $Dest "SHA256SUMS") -Value "$sha  libwelvet_cuda.dll" -Encoding ascii

@(
  "# windows_amd64"
  ""
  "Shipped ``libwelvet_cuda`` for Windows x86-64 (MSVC + NVIDIA CUDA Toolkit)."
  ""
  "| File | Role |"
  "|------|------|"
  "| ``libwelvet_cuda.dll`` | Shared library (runtime) |"
  "| ``libwelvet_cuda.lib`` | Import lib (optional; Go loads the DLL dynamically) |"
  ""
  "Built from welvet.cuda **$Version** (``scripts/build_windows.cmd`` / cmake + MSVC + nvcc)."
  ""
  "Go binding: ``nativecuda`` with ``cgo && windows && amd64``. Put this directory (or CUDA ``bin``) on ``PATH`` so ``cudart`` resolves at runtime, or set ``WELVET_CUDA_LIBRARY`` to the DLL path."
) | Set-Content -Path (Join-Path $Dest "README.md") -Encoding utf8

Write-Host "Wrote MANIFEST + SHA256SUMS for windows_amd64 ($Version)"
