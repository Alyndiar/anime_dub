@echo off
setlocal

REM Point de départ = répertoire actuel
set "ROOT=%cd%"

echo Création de l'arborescence sous %ROOT%
echo.

REM Dossier data
mkdir "%ROOT%\data" 2>nul
mkdir "%ROOT%\data\episodes_raw" 2>nul
mkdir "%ROOT%\data\audio_raw" 2>nul
mkdir "%ROOT%\data\audio_stems" 2>nul
mkdir "%ROOT%\data\diarization" 2>nul

REM Transcripts
mkdir "%ROOT%\data\transcripts" 2>nul
mkdir "%ROOT%\data\transcripts\whisper_json" 2>nul
mkdir "%ROOT%\data\transcripts\zh_srt" 2>nul
mkdir "%ROOT%\data\transcripts\fr_srt" 2>nul

REM Segments + voix + banque + audio final + vidéos finalisées
mkdir "%ROOT%\data\segments" 2>nul
mkdir "%ROOT%\data\voices" 2>nul
mkdir "%ROOT%\data\speaker_bank" 2>nul
mkdir "%ROOT%\data\dub_audio" 2>nul
mkdir "%ROOT%\data\episodes_dubbed" 2>nul

REM Dossier scripts
mkdir "%ROOT%\scripts" 2>nul

REM Dossier config
mkdir "%ROOT%\config" 2>nul

echo.
echo Arborescence créée (ou déjà existante).
echo Termine.
endlocal
