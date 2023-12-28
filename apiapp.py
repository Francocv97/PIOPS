from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
import pandas as pd
import json
import gzip

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

# Usa solo el 10% de los datos
df1 = df1.sample(frac=0.1, random_state=1)

# Crea una matriz de características utilizando TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df1['title'])

# Entrena un modelo NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    # Transforma el título en un vector
    titulo_vec = vectorizer.transform([titulo])

    # Obtiene los índices de los 5 juegos más cercanos
    _, indices = nbrs.kneighbors(titulo_vec)

    # Devuelve los juegos recomendados
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(df1['title'].iloc[indices[0][1:]]))}



# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action