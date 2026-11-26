@echo off
rem The hash tag in the DLL name is sha256(NssaOps.c + NssaOps.h)[:16]
rem (2e46f036e04cf55e today); bump it if the C source changes, mirroring the
rem tag tools/nssa_ops.py computes when it loads this library.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not defined XDG_CACHE_HOME set "XDG_CACHE_HOME=%USERPROFILE%\.cache"
set "OUT=%XDG_CACHE_HOME%\open_health"
if not exist "%OUT%" mkdir "%OUT%"
cl /nologo /O2 /fp:strict /LD apps\ios\OuraApp\NssaOps.c /Fe:"%OUT%\nssa_ops_2e46f036e04cf55e.dll" /Fo:"%OUT%\" /link /EXPORT:nssa_create_windows /EXPORT:nssa_find_indices /EXPORT:nssa_biquad_cascade /EXPORT:nssa_find_peaks /EXPORT:nssa_classifier_predict
