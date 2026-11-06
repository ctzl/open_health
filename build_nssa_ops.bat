@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\Users\root\src\open_health
cl /nologo /O2 /fp:strict /LD apps\ios\OuraApp\NssaOps.c /Fe:C:\Users\root\.cache\open_health\nssa_ops_2e46f036e04cf55e.dll /Fo:C:\Users\root\.cache\open_health\ /link /EXPORT:nssa_create_windows /EXPORT:nssa_find_indices /EXPORT:nssa_biquad_cascade /EXPORT:nssa_find_peaks /EXPORT:nssa_classifier_predict
