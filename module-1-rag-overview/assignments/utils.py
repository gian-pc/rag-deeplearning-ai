import json
import numpy as np
import pandas as pd
from dateutil import parser
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargamos el mismo modelo que usó el curso para calcular los embeddings
# Esto garantiza que nuestras búsquedas sean compatibles con embeddings.joblib
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Cargamos los embeddings pre-calculados (870 noticias x 768 dimensiones)
EMBEDDINGS = joblib.load("embeddings.joblib")

# Cargamos el dataset de noticias
NEWS_DATA = pd.read_csv("news_data_dedup.csv").to_dict(orient='records')

def pprint(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))

def format_date(date_string):
    date_object = parser.parse(date_string)
    return date_object.strftime("%Y-%m-%d")

def read_dataframe(path):
    df = pd.read_csv(path)
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)
    return df.to_dict(orient='records')

def retrieve(query, top_k=5):
    # Convertimos la query al mismo formato que los embeddings pre-calculados
    query_embedding = model.encode(query)
    
    # Comparamos contra los 870 embeddings pre-calculados
    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1), EMBEDDINGS
    )[0]
    
    # Ordenamos de mayor a menor similitud
    similarity_indices = np.argsort(-similarity_scores)
    
    # Devolvemos los top_k índices más relevantes
    return similarity_indices[:top_k]

def generate_with_single_input(prompt, max_tokens=1024, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return {
        "role": "assistant",
        "content": response.choices[0].message.content
    }
