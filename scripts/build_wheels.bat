@echo off
rem Ricompila i moduli CUDA allegati alle release (tiny-cuda-nn e moduli del codice Inria) e ne
rem salva i wheel in tools\wheels. Serve solo a chi prepara una release: chi installa il programma
rem riceve i wheel gia' compilati.
rem
rem Richiede: Visual Studio con gli strumenti C++ (MSVC 14.4x), CUDA Toolkit 12.4, Git, e
rem l'ambiente .venv gia' creato da Installa.bat. Adattare i due percorsi qui sotto se diversi.
setlocal enabledelayedexpansion
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

rem Combinazione verificata: MSVC 14.44 con CUDA 12.4 (con CUDA 11.8 questo compilatore viene rifiutato).
call "%VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
rem Architetture delle schede RTX serie 20, 30 e 40 (e delle professionali corrispondenti).
set TCNN_CUDA_ARCHITECTURES=75;86;89
rem Usa il compilatore dell'ambiente appena preparato, non quello cercato da setuptools.
set DISTUTILS_USE_SDK=1
set MSSdk=1
rem tiny-cuda-nn produce file oggetto oltre il limite predefinito del compilatore Microsoft.
set NVCC_APPEND_FLAGS=-Xcompiler /bigobj

cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist tools\gaussian-splatting\submodules\simple-knn\setup.py (
  echo Serve il codice Inria completo di sottomoduli:
  echo   git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting tools\gaussian-splatting
  exit /b 1
)
rem --no-build-isolation: gli script di build usano pkg_resources, assente nelle versioni recenti di setuptools.
%PY% -m pip install ninja
%PY% -m pip wheel --no-build-isolation --no-deps -w tools\wheels "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
if errorlevel 1 exit /b 1
for %%m in (diff-gaussian-rasterization simple-knn fused-ssim) do (
  %PY% -m pip wheel --no-build-isolation --no-deps -w tools\wheels tools\gaussian-splatting\submodules\%%m
  if errorlevel 1 exit /b 1
)
dir /b tools\wheels
