@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: Run this script from Command propmt (PowerShell not supported) 
:: to install napari-annotatorj on your Windows PC
:: after you have successfully installed Python 3.7
:: or later

:: to verify Python install, run the next command
python -V
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error checking Python version. Do you have Python installed on your PC?"
    exit /B 1
)
echo "Verified Python version"

:: current folder
set root_dir=%~dp0

:: parent of current folder
for %%i in ("%~dp0..") do set "parent_folder=%%~fi"
:: name of the virtual environment
set venvName=%parent_folder%\napariAnnotatorjEnv

:: check if the default virtenv already exists in the
:: expected location
if exist %venvName%\ (
  if exist %venvName%\Lib\site-packages\pip\ (
  	echo Virtenv %venvName% already exists
	goto :VIRTENVREADY
  ) else (
  	echo Virtenv %venvName% does not exist, please run install.bat first
    exit /b 1
  )
) else (
  echo Virtenv %venvName% does not exist, please run install.bat first
  exit /b 1
)

:VIRTENVREADY

:: check if run from cmd or powershell
echo %PSModulePath% | findstr %USERPROFILE% >NUL
IF %ERRORLEVEL% EQU 0 goto :ISPOWERSHELL

echo Running from Command prompt 
goto :ISCMD

:ISPOWERSHELL
echo Error: Cannot run from PowerShell. Please run this .bat file from Command prompt
exit /b 1


:ISCMD
:: activate virtenv
start cmd /k "%venvName%\Scripts\activate && echo 'Activated virtual environment successfully: ' %venvName% && napari"

