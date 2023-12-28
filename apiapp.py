import json
import pandas as pd
import gzip
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()

archivo = 'output_steam_games_limpio_reducido.json.gz'

def read_json_gzip(file_path):
    with gzip.open(file_path, 'r') as file:
        data = file.read().decode('utf-8')
        if data[0] == '[':
            for item in json.loads(data):
                yield item
        else:
            for line in data.splitlines():
                yield json.loads(line)

# Lista para guardar cada fila
lista = list(read_json_gzip(archivo))

# Crea un DataFrame a partir de la lista
df1 = pd.DataFrame(lista)

# Elimina las filas con valores NaN en la columna de juegos
df1 = df1.dropna(subset=['title'])

# Crea una matriz de características utilizando CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df1['title'])

# Crea una serie para mapear los índices de los juegos a sus títulos
indices_titulo = pd.Series(df1.index, index=df1['title']).drop_duplicates()

# Libera memoria no utilizada
del lista

cosine_sim_titulo = cosine_similarity(count_matrix, count_matrix)

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    idx = indices_titulo[titulo]
    sim_scores = np.argsort(cosine_sim_titulo[idx])[::-1][1:6]
    juegos_recomendados = df1['title'].iloc[sim_scores]
    
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(juegos_recomendados))}



# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action