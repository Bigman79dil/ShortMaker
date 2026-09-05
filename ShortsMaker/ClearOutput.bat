@echo off
cd /d "%~dp0"
if exist "output_shorts" (
    cd "output_shorts"
    del /q/f/s *.*
    for /d %%i in (*) do rd /s /q "%%i"
    echo "output_shorts" cleared successfully.
) else (
    echo Error: "output_shorts" folder not found next to this script.
)
pause
