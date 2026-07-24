@echo off
setlocal
pushd "%~dp0"

echo [1/4] Checking Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.12 was not found.
  echo Run: py -0p
  echo Then confirm that Python 3.12 is installed.
  pause
  popd
  exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv ".venv"
  if errorlevel 1 goto :error
) else (
  echo Virtual environment already exists.
)

echo [3/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [4/4] Installing packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup completed successfully.
echo Double-click run_app.bat to start the application.
pause
popd
exit /b 0

:error
echo.
echo ERROR: Setup failed. Review the messages above.
pause
popd
exit /b 1
