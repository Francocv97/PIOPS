import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI
from starlette.testclient import TestClient

# Ruta a tu archivo JSON
archivo = 'output_steam_games_limpio.json'

# Lista para guardar cada fila
lista = []

with open(archivo, 'r', encoding='utf-8') as file:
    data = file.read()
    if data[0] == '[':
        # Los datos están en formato de array
        lista = json.loads(data)
    else:
        # Los datos están separados por nuevas líneas
        for line in data.splitlines():
            lista.append(json.loads(line))

# Crear un DataFrame a partir de la lista
df = pd.DataFrame(lista)

# Asegúrate de que 'release_date' sea un objeto datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce', format='%Y-%m-%d')

# Calcular las horas desde el lanzamiento hasta la fecha actual
now = datetime.now()
df['hours_since_release'] = (now - df['release_date']).dt.total_seconds() / 3600

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str) -> Dict[str, int]:
    # Asegúrate de que todos los valores en 'genres' sean listas
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar el DataFrame por el género especificado
    df_genre = df[df['genres'].apply(lambda x: genero in x)]
    
    # Agrupar por año de lanzamiento y sumar las horas desde el lanzamiento
    df_genre_year = df_genre.groupby(df_genre['release_date'].dt.year)['hours_since_release'].sum()
    
    # Encontrar el año de lanzamiento con más horas desde el lanzamiento
    max_year = df_genre_year.idxmax()
    
    return {f"Año de lanzamiento con más horas jugadas para {genero}" : max_year}

# Crear un cliente de prueba
client = TestClient(app)

#http://localhost:8000/PlayTimeGenre/Action

#http://127.0.0.1:8000/PlayTimeGenre/Casual

#uvicorn apiapp:app --reload
