@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "IMGUI=%~dp0vendor\imgui"
set "VENDOR=%~dp0vendor"
cl /nologo /std:c++17 /O2 /MT /EHsc /DUNICODE /D_UNICODE /I"%VENDOR%" /I"%IMGUI%" /I"%IMGUI%\backends" pb2cast.cpp "%IMGUI%\imgui.cpp" "%IMGUI%\imgui_draw.cpp" "%IMGUI%\imgui_tables.cpp" "%IMGUI%\imgui_widgets.cpp" "%IMGUI%\backends\imgui_impl_win32.cpp" "%IMGUI%\backends\imgui_impl_opengl2.cpp" /Fe:"..\pb2cast_v07.exe" /link /SUBSYSTEM:WINDOWS opengl32.lib user32.lib gdi32.lib shell32.lib ole32.lib dwmapi.lib





