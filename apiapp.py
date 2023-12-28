import json
import pandas as pd
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

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

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    idx = indices_titulo[titulo]
    sim_scores = list(enumerate(cosine_sim_titulo[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    juego_indices = [i[0] for i in sim_scores]
    juegos_recomendados = df1['title'].iloc[juego_indices]
    
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(juegos_recomendados))}



# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action