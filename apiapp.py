from fastapi import FastAPI
from typing import Dict
import pandas as pd
from datetime import datetime
import uvicorn

app = FastAPI()

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('1rafuncion.csv')

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str) -> Dict[str, int]:
    # Asegúrate de que todos los valores en 'genres' sean listas
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])

    # Filtrar el DataFrame por el género especificado
    df_genre = df[df['genres'].apply(lambda x: genero in x)]

    # Verificar si hay datos para el género
    if df_genre.empty:
        return {"mensaje": f"No hay datos para el género {genero}"}

    # Agrupar por año de lanzamiento y sumar las horas desde el lanzamiento
    df_genre_year = df_genre.groupby(df_genre['release_date'].dt.year)['hours_since_release'].sum()

    # Verificar si hay datos para calcular el índice máximo
    if df_genre_year.empty:
        return {"mensaje": f"No hay datos para calcular el año de lanzamiento con más horas jugadas para {genero}"}

    # Encontrar el año de lanzamiento con más horas desde el lanzamiento
    max_year = df_genre_year.idxmax()

    return {f"Año de lanzamiento con más horas jugadas para {genero}": max_year}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


#http://127.0.0.1:8000/PlayTimeGenre/Casual
#uvicorn apiapp:app --reload