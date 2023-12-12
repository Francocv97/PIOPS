import nest_asyncio
from fastapi import FastAPI
from starlette.testclient import TestClient
from typing import Dict, Union
import pandas as pd
from datetime import datetime
import uvicorn

# Esto permite a Jupyter ejecutar código asíncrono
nest_asyncio.apply()

# Ruta a tu archivo CSV
archivo = 'output_steam_games.csv'

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(archivo)

# Asegúrate de que 'release_date' sea un objeto datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce', format='%Y-%m-%d')

# Calcular las horas desde el lanzamiento hasta la fecha actual
now = datetime.now()
df['hours_since_release'] = (now - df['release_date']).dt.total_seconds() / 3600

app = FastAPI()

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str) -> Dict[str, Union[str, int]]:
    # Asegúrate de que todos los valores en 'genres' sean listas
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar el DataFrame por el género especificado
    df_genre = df[df['genres'].apply(lambda x: genero in x)]
    
    # Si no hay datos para el género especificado, devuelve un mensaje de error
    if df_genre.empty:
        return {f"mensaje" : f"No hay datos para el género {genero}", "Año de lanzamiento con más horas jugadas para {genero}" : -1}
    
    # Agrupar por año de lanzamiento y sumar las horas desde el lanzamiento
    df_genre_year = df_genre.groupby(df_genre['release_date'].dt.year)['hours_since_release'].sum()
    
    # Encontrar el año de lanzamiento con más horas desde el lanzamiento
    max_year = df_genre_year.idxmax()
    
    return {f"Año de lanzamiento con más horas jugadas para {genero}" : max_year}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




#http://localhost:8000/PlayTimeGenre/Action

#http://127.0.0.1:8000/PlayTimeGenre/Casual

#uvicorn apiapp:app --reload