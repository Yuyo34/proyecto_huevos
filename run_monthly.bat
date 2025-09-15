@echo off
cd /d "%~dp0"
python ".\pipeline_monthly.py" --config ".\config.yml"
 pause >nul
