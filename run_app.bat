@echo off
setlocal
pushd "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment was not found.
  echo Run setup_windows.bat first.
  pause
  popd
  exit /b 1
)

echo Starting Horse Racing AI...
".venv\Scripts\python.exe" -m streamlit run app.py

if errorlevel 1 (
  echo.
  echo ERROR: The application stopped with an error.
  pause
)

popd
endlocal
