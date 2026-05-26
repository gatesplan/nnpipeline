@echo off
:: Check admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Restarting with admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"

:: Run in Windows Terminal with PowerShell
wt.exe -p "PowerShell" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0start-claude.ps1" -WorkDir "%~dp0"
