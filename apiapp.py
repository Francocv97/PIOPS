import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI
from starlette.testclient import TestClient
import numpy as np


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

# Supongamos que 'df' es tu DataFrame
df = pd.read_csv('2dafuncion.csv')

# Limpia los valores NaN en la columna de géneros
df['genres'] = df['genres'].replace(np.nan, '')

@app.get("/UserForGenre/{genero}")
async def read_user_for_genre(genero: str):
    # Filtramos el dataframe por el género dado
    df_genre = df[df['genres'].str.contains(genero)]
    
    # Agrupamos por usuario y sumamos las horas jugadas
    df_grouped = df_genre.groupby('user_id')['playtime_forever'].sum().reset_index()
    
    # Encontramos el usuario con más horas jugadas
    max_playtime_user = df_grouped[df_grouped['playtime_forever'] == df_grouped['playtime_forever'].max()]['user_id'].values[0]
    
    # Creamos un dataframe con las horas jugadas por año
    df_genre['year'] = pd.to_datetime(df_genre['posted_date']).dt.year
    playtime_per_year = df_genre.groupby('year')['playtime_forever'].sum().reset_index().to_dict('records')
    
    return {"Usuario con más horas jugadas para Género {}".format(genero) : max_playtime_user, "Horas jugadas": playtime_per_year}
