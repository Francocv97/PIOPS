import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI
from starlette.testclient import TestClient
import numpy as np
from collections import Counter
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Ruta a tu archivo JSON comprimido
archivo = 'output_steam_games_limpio_reducido.json.gz'

# Lista para guardar cada fila
lista = []

with gzip.open(archivo, 'r') as file:
    data = file.read().decode('utf-8')
    if data[0] == '[':
        # Los datos están en formato de array
        lista = json.loads(data)
    else:
        # Los datos están separados por nuevas líneas
        for line in data.splitlines():
            lista.append(json.loads(line))

# Crear un DataFrame a partir de la lista
df1 = pd.DataFrame(lista)

# Elimina las filas con valores NaN en la columna de juegos
df1 = df1.dropna(subset=['title'])

# Crea una matriz de características utilizando CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df1['title'])

# Calcula la similitud del coseno
cosine_sim_titulo = cosine_similarity(count_matrix, count_matrix)

# Crea una serie para mapear los índices de los juegos a sus títulos
indices_titulo = pd.Series(df1.index, index=df1['title']).drop_duplicates()

def recomendacion_juego(titulo, cosine_sim=cosine_sim_titulo):
    # Obtiene el índice del juego que coincide con el título
    idx = indices_titulo[titulo]

    # Obtiene las puntuaciones de similitud por pares de todos los juegos con ese juego
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordena los juegos en función de las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtiene las puntuaciones de los 5 juegos más similares
    sim_scores = sim_scores[1:6]

    # Obtiene los índices de los juegos
    juego_indices = [i[0] for i in sim_scores]

    # Devuelve los 5 juegos más similares
    return df1['title'].iloc[juego_indices]

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    # Llama a la función de recomendación de juego
    juegos_recomendados = recomendacion_juego(titulo)

    # Devuelve los juegos recomendados
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(juegos_recomendados))}

# Carga los datos para el segundo modelo de recomendación
df_usuario = pd.read_csv('2dafuncion_final.csv')

# Crea la matriz de utilidad usando 'playtime_forever' como valor
utilidad = df_usuario.pivot_table(index='user_id', columns='title', values='playtime_forever')

# Calcula la similitud del coseno
similitud_usuario = cosine_similarity(utilidad.fillna(0))

# Crea un mapeo de ID de usuario a índice de matriz
user_id_to_index = {user_id: index for index, user_id in enumerate(utilidad.index)}

def recomendacion_usuario(user_id):
    # Obtiene el índice de la matriz para el ID de usuario
    user_index = user_id_to_index[user_id]
    # Obtiene los índices de los 6 usuarios más similares
    indices_similares = np.argsort(similitud_usuario[user_index])[-7:-1][::-1]
    # Encuentra los juegos que les gustaron a los usuarios similares
    juegos_recomendados = utilidad.iloc[indices_similares].mean().sort_values(ascending=False).index[:6]
    # Excluye el primer juego (el más recomendado) y devuelve los siguientes 5
    return juegos_recomendados[1:]

@app.get("/recomendacion_usuario/{user_id}")
def get_recomendacion_usuario(user_id: str):
    if user_id in user_id_to_index:
        # Llama a la función de recomendación de usuario
        juegos_recomendados = recomendacion_usuario(user_id)

        # Devuelve los juegos recomendados
        return {"Juegos recomendados para el usuario {}: {}".format(user_id, list(juegos_recomendados))}
    else:
        return {"error": "El ID de usuario {} no se encuentra en los datos.".format(user_id)}
    
    # Elimina las filas con valores NaN en la columna de juegos
df1 = df1.dropna(subset=['title'])

# Crea una matriz de características utilizando CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df1['title'])

# Calcula la similitud del coseno
cosine_sim_titulo = cosine_similarity(count_matrix, count_matrix)

# Crea una serie para mapear los índices de los juegos a sus títulos
indices_titulo = pd.Series(df1.index, index=df1['title']).drop_duplicates()

def recomendacion_juego(titulo, cosine_sim=cosine_sim_titulo):
    # Obtiene el índice del juego que coincide con el título
    idx = indices_titulo[titulo]

    # Obtiene las puntuaciones de similitud por pares de todos los juegos con ese juego
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordena los juegos en función de las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtiene las puntuaciones de los 5 juegos más similares
    sim_scores = sim_scores[1:6]

    # Obtiene los índices de los juegos
    juego_indices = [i[0] for i in sim_scores]

    # Devuelve los 5 juegos más similares
    return df1['title'].iloc[juego_indices]

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    # Llama a la función de recomendación de juego
    juegos_recomendados = recomendacion_juego(titulo)

    # Devuelve los juegos recomendados
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(juegos_recomendados))}

# Carga los datos para el segundo modelo de recomendación
df_usuario = pd.read_csv('2dafuncion_final.csv')

# Crea la matriz de utilidad usando 'playtime_forever' como valor
utilidad = df_usuario.pivot_table(index='user_id', columns='title', values='playtime_forever')

# Calcula la similitud del coseno
similitud_usuario = cosine_similarity(utilidad.fillna(0))

# Crea un mapeo de ID de usuario a índice de matriz
user_id_to_index = {user_id: index for index, user_id in enumerate(utilidad.index)}

def recomendacion_usuario(user_id):
    # Obtiene el índice de la matriz para el ID de usuario
    user_index = user_id_to_index[user_id]
    # Obtiene los índices de los 6 usuarios más similares
    indices_similares = np.argsort(similitud_usuario[user_index])[-7:-1][::-1]
    # Encuentra los juegos que les gustaron a los usuarios similares
    juegos_recomendados = utilidad.iloc[indices_similares].mean().sort_values(ascending=False).index[:6]
    # Excluye el primer juego (el más recomendado) y devuelve los siguientes 5
    return juegos_recomendados[1:]

@app.get("/recomendacion_usuario/{user_id}")
def get_recomendacion_usuario(user_id: str):
    if user_id in user_id_to_index:
        # Llama a la función de recomendación de usuario
        juegos_recomendados = recomendacion_usuario(user_id)

        # Devuelve los juegos recomendados
        return {"Juegos recomendados para el usuario {}: {}".format(user_id, list(juegos_recomendados))}
    else:
        return {"error": "El ID de usuario {} no se encuentra en los datos.".format(user_id)}



# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action