@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "configs\so101_device_calibration.json" if "%SO101_CALIBRATION_PATH%"=="" (
  echo 请先放入本机 SO-101 标定文件 configs\so101_device_calibration.json
  pause
  exit /b 1
)
python examples\so101_xyz_ui.py
if errorlevel 1 pause
