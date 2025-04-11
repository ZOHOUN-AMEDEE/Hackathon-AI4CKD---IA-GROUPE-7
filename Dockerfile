FROM python:3.9-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application et le modèle
COPY app.py .
COPY best_smote_gradient_boosting.pkl .

# Exposer le port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
