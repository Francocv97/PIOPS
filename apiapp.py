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


# Ruta a tu archivo JSON comprimido
archivo = 'output_steam_games_limpio.json.gz'

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

# Descomprimir el archivo csv
with gzip.open('df_funcion3.csv.gz', 'rt') as f:
    df_funcion3 = pd.read_csv(f)

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


# Cargar el DataFrame desde el archivo CSV
df_userid_recommend = pd.read_csv('df_userid_recommend.csv')

@app.get("/Sentiment_analysis/{year}")
def read_sentiment_analysis(year: int):
    df_year = df_userid_recommend[df_userid_recommend['year'] == year]
    sentiment_counts = df_year['sentiment_analysis'].value_counts().to_dict()
    sentiment_dict = {'Negative': sentiment_counts.get(0, 0), 'Neutral': sentiment_counts.get(1, 0), 'Positive': sentiment_counts.get(2, 0)}
    return sentiment_dict


# Carga los datos para el primer modelo de recomendación
df_titulo = pd.read_csv('output_steam_games_final.csv')

# Elimina las filas con valores NaN en la columna de juegos
df_titulo = df_titulo.dropna(subset=['title'])

# Crea una matriz de características utilizando CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df_titulo['title'])

# Calcula la similitud del coseno
cosine_sim_titulo = cosine_similarity(count_matrix, count_matrix)

# Crea una serie para mapear los índices de los juegos a sus títulos
indices_titulo = pd.Series(df_titulo.index, index=df_titulo['title']).drop_duplicates()

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
    return df_titulo['title'].iloc[juego_indices]

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