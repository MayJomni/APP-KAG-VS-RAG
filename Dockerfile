FROM python:3.12-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install uv

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock* ./

# Installer les dépendances Python
RUN uv sync --frozen --no-dev

# Copier le code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p results mlruns

# Port de l'application
EXPOSE 8000

# Lancement du serveur
CMD ["uv", "run", "python", "kag_server.py"]
