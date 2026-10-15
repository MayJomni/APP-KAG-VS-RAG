@echo off
title Lancement Plateforme RAG vs KAG - Ooredoo AI Lab
color 0B
echo ==============================================================================
echo       OOREDOO AI LAB - DEMARRAGE DU SERVEUR FASTAPI & DASHBOARD WEB
echo ==============================================================================
echo.
echo 1. Ouverture du navigateur sur http://localhost:8000...
start "" "http://localhost:8000"
echo 2. Lancement du serveur backend Uvicorn...
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn fast_server:app --host 127.0.0.1 --port 8000
pause
