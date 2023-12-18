import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI
from starlette.testclient import TestClient
import numpy as np
from collections import Counter

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
df1 = pd.DataFrame(lista)

# Verificar si 'release_date' existe en df1
if 'release_date' in df1.columns:
    # Asegúrate de que 'release_date' sea un objeto datetime
    df1['release_date'] = pd.to_datetime(df1['release_date'], errors='coerce', format='%Y-%m-%d')

    # Calcular las horas desde el lanzamiento hasta la fecha actual
    now = datetime.now()
    df1['hours_since_release'] = (now - df1['release_date']).dt.total_seconds() / 3600

# Supongamos que 'df' es tu DataFrame
df2 = pd.read_csv('2dafuncion.csv')

# Verificar si 'release_date' existe en df2
if 'release_date' in df2.columns:
    # Asegúrate de que 'release_date' sea un objeto datetime
    df2['release_date'] = pd.to_datetime(df2['release_date'], errors='coerce', format='%Y-%m-%d')

    # Calcular las horas desde el lanzamiento hasta la fecha actual
    now = datetime.now()
    df2['hours_since_release'] = (now - df2['release_date']).dt.total_seconds() / 3600

# Limpia los valores NaN en la columna de géneros
df2['genres'] = df2['genres'].replace(np.nan, '')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str) -> Dict[str, int]:
    # Asegúrate de que todos los valores en 'genres' sean listas
    df1['genres'] = df1['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar el DataFrame por el género especificado
    df_genre = df1[df1['genres'].apply(lambda x: genero in x)]
    
    # Agrupar por año de lanzamiento y sumar las horas desde el lanzamiento
    df_genre_year = df_genre.groupby(df_genre['release_date'].dt.year)['hours_since_release'].sum()
    
    # Encontrar el año de lanzamiento con más horas desde el lanzamiento
    max_year = df_genre_year.idxmax()
    
    return {f"Año de lanzamiento con más horas jugadas para {genero}" : max_year}

@app.get("/UserForGenre/{genero}")
async def read_user_for_genre(genero: str):
    # Filtramos el dataframe por el género dado
    df_genre = df2[df2['genres'].str.contains(genero)]
    
    # Agrupamos por usuario y sumamos las horas jugadas
    df_grouped = df_genre.groupby('user_id')['playtime_forever'].sum().reset_index()
    
    # Encontramos el usuario con más horas jugadas
    max_playtime_user = df_grouped[df_grouped['playtime_forever'] == df_grouped['playtime_forever'].max()]['user_id'].values[0]
    
    # Creamos un dataframe con las horas jugadas por año
    df_genre['year'] = pd.to_datetime(df_genre['posted_date']).dt.year
    playtime_per_year = df_genre.groupby('year')['playtime_forever'].sum().reset_index().to_dict('records')
    
    return {"Usuario con más horas jugadas para Género {}".format(genero) : max_playtime_user, "Horas jugadas": playtime_per_year}

# Cargar el archivo csv
df_funcion3 = pd.read_csv('df_funcion3.csv')

@app.get("/UsersRecommend/{year}")
async def get_recommend(year: int):
    # Crear un contador para los juegos
    game_counter = Counter()

    # Convertir la columna 'posted' a datetime
    df_funcion3['posted_date'] = pd.to_datetime(df_funcion3['posted'].str[7:-1], errors='coerce')

    # Filtrar las filas donde 'posted' es del año dado y 'recommend' es True
    df_filtered = df_funcion3[(df_funcion3['posted_date'].dt.year == year) & df_funcion3['recommend']]

    # Contar las ocurrencias de cada juego
    game_counter = Counter(df_filtered['item_name'])

    # Obtener los 3 juegos más comunes
    most_common_games = game_counter.most_common(3)

    # Devolver los resultados
    return [{"Puesto " + str(i+1) : most_common_games[i]} if len(most_common_games) > i else {"Puesto " + str(i+1) : None} for i in range(3)]

@app.get("/UsersNotRecommend/{year}")
async def get_not_recommend(year: int):
    # Crear un contador para los juegos
    game_counter = Counter()

    # Convertir la columna 'posted' a datetime
    df_funcion3['posted_date'] = pd.to_datetime(df_funcion3['posted'].str[7:-1], errors='coerce')

    # Filtrar las filas donde 'posted' es del año dado y 'recommend' es False
    df_filtered = df_funcion3[(df_funcion3['posted_date'].dt.year == year) & (df_funcion3['recommend'] == False)]

    # Contar las ocurrencias de cada juego
    game_counter = Counter(df_filtered['item_name'])

    # Obtener los 3 juegos más comunes
    most_common_games = game_counter.most_common(3)

    # Devolver los resultados
    return [{"Puesto " + str(i+1) : most_common_games[i]} if len(most_common_games) > i else {"Puesto " + str(i+1) : None} for i in range(3)]



# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action