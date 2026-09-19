@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars32.bat" >nul
cl /nologo /EHsc /MT animdecode.cpp /Fe:animdecode.exe
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --onedir --name pb2cast-worker --distpath ..\bin --workpath build --specpath build worker.py
if errorlevel 1 exit /b 1
copy /y animdecode.exe ..\bin\pb2cast-worker\animdecode.exe
