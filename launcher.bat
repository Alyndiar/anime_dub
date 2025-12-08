@echo off
REM Lance l'interface graphique du pipeline depuis la racine du projet
setlocal
pushd "%~dp0"
python "scripts\gui_pipeline.py" %*
popd
endlocal
