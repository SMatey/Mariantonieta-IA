"""
API de Películas - Sistema de Recomendación usando KNN
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Optional
from ..models.movies_api_models import MovieRecommendationRequest, MovieRecommendationResponse, MovieRatingRequest, MovieRatingResponse

# Importar constantes desde el paquete api
from .. import constants as const

app = FastAPI(title="Movies Recommendation API", version="1.0.0")


# Variables globales para el modelo y datos
_loaded_model_data = None
_movies_data = None
_ratings_data = None

def load_movies_model_and_data():
    """Carga el modelo KNN y los datos de películas"""
    global _loaded_model_data, _movies_data, _ratings_data
    
    if _loaded_model_data is not None:
        return _loaded_model_data, _movies_data, _ratings_data
    
    try:
        # Cargar modelo KNN
        model_path = os.path.join(const.BASE_DIR, 'ml_models', 'knn_movie_recommendation_model.pkl')
        with open(model_path, 'rb') as f:
            knn_model = pickle.load(f)
        
        # Cargar datos de películas (simulados para el ejemplo)
        # En un caso real, cargarías desde archivos CSV
        _movies_data = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'Toy Story (1995)',
                'Jumanji (1995)', 
                'Grumpier Old Men (1995)',
                'Waiting to Exhale (1995)',
                'Father of the Bride Part II (1995)',
                'Heat (1995)',
                'Sabrina (1995)',
                'Tom and Huck (1995)',
                'Sudden Death (1995)',
                'GoldenEye (1995)'
            ],
            'genres': [
                'Adventure|Animation|Children|Comedy|Fantasy',
                'Adventure|Children|Fantasy',
                'Comedy|Romance',
                'Comedy|Drama|Romance',
                'Comedy',
                'Action|Crime|Thriller',
                'Comedy|Romance',
                'Adventure|Children',
                'Action',
                'Action|Adventure|Thriller'
            ]
        })
        
        # Datos de ratings simulados
        _ratings_data = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 3, 3, 3],
            'movieId': [1, 2, 3, 1, 4, 2, 3, 5],
            'rating': [4.0, 3.5, 4.5, 5.0, 3.0, 4.0, 4.5, 3.5]
        })
        
        _loaded_model_data = {
            'model': knn_model,
            'model_info': {
                'type': 'K-Nearest Neighbors',
                'algorithm': 'cosine similarity',
                'n_neighbors': getattr(knn_model, 'n_neighbors', 11)
            }
        }
        
        return _loaded_model_data, _movies_data, _ratings_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

def get_movie_recommendations_by_similarity(movie_id, num_recommendations=5):
    """Obtiene recomendaciones basadas en similitud de películas"""
    model_data, movies_df, ratings_df = load_movies_model_and_data()
    
    # Buscar película
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if movie_info.empty:
        return []
    
    movie_title = movie_info.iloc[0]['title']
    movie_genres = movie_info.iloc[0]['genres'].split('|')
    
    # Encontrar películas similares por género
    similar_movies = []
    for _, movie in movies_df.iterrows():
        if movie['movieId'] != movie_id:
            movie_genres_list = movie['genres'].split('|')
            # Calcular similitud de géneros
            common_genres = set(movie_genres).intersection(set(movie_genres_list))
            similarity = len(common_genres) / len(set(movie_genres).union(set(movie_genres_list)))
            
            if similarity > 0:
                # Obtener rating promedio
                movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]
                avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 3.5
                
                similar_movies.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'similarity': similarity,
                    'avg_rating': avg_rating,
                    'common_genres': list(common_genres)
                })
    
    # Ordenar por similitud y rating
    similar_movies.sort(key=lambda x: (x['similarity'], x['avg_rating']), reverse=True)
    
    return similar_movies[:num_recommendations]

def get_user_recommendations(user_id, num_recommendations=5):
    """Obtiene recomendaciones para un usuario específico"""
    model_data, movies_df, ratings_df = load_movies_model_and_data()
    
    # Obtener las películas que el usuario ha calificado positivamente (>= 4.0)
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return []
    
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
    
    # Encontrar géneros preferidos del usuario
    preferred_genres = []
    for movie_id in liked_movies:
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            genres = movie_info.iloc[0]['genres'].split('|')
            preferred_genres.extend(genres)
    
    # Contar géneros más frecuentes
    from collections import Counter
    genre_counts = Counter(preferred_genres)
    top_genres = [genre for genre, count in genre_counts.most_common(3)]
    
    # Recomendar películas no vistas con géneros similares
    unrated_movies = movies_df[~movies_df['movieId'].isin(user_ratings['movieId'])]
    recommendations = []
    
    for _, movie in unrated_movies.iterrows():
        movie_genres = movie['genres'].split('|')
        genre_match = any(genre in top_genres for genre in movie_genres)
        
        if genre_match:
            # Calcular score basado en rating promedio
            movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]
            avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 3.5
            
            # Calcular similitud con géneros preferidos
            common_genres = set(top_genres).intersection(set(movie_genres))
            genre_similarity = len(common_genres) / len(top_genres) if top_genres else 0
            
            recommendations.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'predicted_rating': avg_rating + (genre_similarity * 0.5),
                'avg_rating': avg_rating,
                'genre_match': list(common_genres)
            })
    
    # Ordenar por predicción de rating
    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    return recommendations[:num_recommendations]

def predict_user_rating(user_id, movie_id):
    """Predice el rating que un usuario daría a una película"""
    model_data, movies_df, ratings_df = load_movies_model_and_data()
    
    # Obtener ratings del usuario
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_avg = user_ratings['rating'].mean() if not user_ratings.empty else 3.5
    
    # Obtener info de la película
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if movie_info.empty:
        return user_avg, 50.0
    
    # Calcular rating promedio de la película
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
    movie_avg = movie_ratings['rating'].mean() if not movie_ratings.empty else 3.5
    
    # Predicción simple basada en promedios y preferencias del usuario
    prediction = (user_avg * 0.6) + (movie_avg * 0.4)
    
    # Confianza basada en datos disponibles
    confidence = 60 + min(30, len(user_ratings) * 3) + min(10, len(movie_ratings))
    
    return prediction, confidence

@app.get("/")
def root():
    return {"message": "Movies Recommendation API", "version": "1.0.0"}

@app.post("/models/movies/recommend", response_model=MovieRecommendationResponse)
def recommend_movies(request: MovieRecommendationRequest):
    """
    Recomienda películas basadas en similitud o preferencias del usuario
    """
    try:
        num_recs = request.num_recommendations or 5
        recommendations = []
        
        if request.movie_id:
            # Recomendaciones basadas en película específica
            recommendations = get_movie_recommendations_by_similarity(request.movie_id, num_recs)
            recommendation_type = f"películas similares a ID {request.movie_id}"
            
        elif request.user_id:
            # Recomendaciones para usuario específico
            recommendations = get_user_recommendations(request.user_id, num_recs)
            recommendation_type = f"recomendaciones personalizadas para usuario {request.user_id}"
            
        else:
            # Recomendaciones generales (películas mejor calificadas)
            model_data, movies_df, ratings_df = load_movies_model_and_data()
            movie_scores = []
            
            for _, movie in movies_df.iterrows():
                movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]
                avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 3.5
                rating_count = len(movie_ratings)
                
                # Score combinado de rating y popularidad
                score = avg_rating + (rating_count * 0.1)
                
                movie_scores.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'avg_rating': avg_rating,
                    'rating_count': rating_count,
                    'score': score
                })
            
            movie_scores.sort(key=lambda x: x['score'], reverse=True)
            recommendations = movie_scores[:num_recs]
            recommendation_type = "películas mejor calificadas"
        
        # Crear interpretación
        if recommendations:
            top_movie = recommendations[0]
            interpretation = (
                f"🎬 RECOMENDACIONES DE PELÍCULAS\n"
                f"Tipo: {recommendation_type}\n"
                f"Películas encontradas: {len(recommendations)}\n\n"
                f"🏆 Mejor recomendación:\n"
                f"Título: {top_movie['title']}\n"
                f"Géneros: {top_movie.get('genres', 'N/A')}\n"
            )
            
            if 'avg_rating' in top_movie:
                interpretation += f"Rating promedio: {top_movie['avg_rating']:.1f}/5.0\n"
            
            if len(recommendations) > 1:
                interpretation += f"\nOtras recomendaciones: {', '.join([r['title'] for r in recommendations[1:3]])}"
                if len(recommendations) > 3:
                    interpretation += "..."
        else:
            interpretation = "❌ No se encontraron recomendaciones con los criterios especificados."
        
        model_data, _, _ = load_movies_model_and_data()
        
        return MovieRecommendationResponse(
            recommendations=recommendations,
            model_info={
                "model_type": model_data['model_info']['type'],
                "algorithm": model_data['model_info']['algorithm'],
                "recommendations_count": len(recommendations),
                "recommendation_type": recommendation_type
            },
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/models/movies/predict-rating", response_model=MovieRatingResponse)
def predict_movie_rating(request: MovieRatingRequest):
    """
    Predice el rating que un usuario daría a una película específica
    """
    try:
        prediction, confidence = predict_user_rating(request.user_id, request.movie_id)
        
        # Obtener información de la película
        model_data, movies_df, ratings_df = load_movies_model_and_data()
        movie_info = movies_df[movies_df['movieId'] == request.movie_id]
        
        if not movie_info.empty:
            movie_title = movie_info.iloc[0]['title']
            movie_genres = movie_info.iloc[0]['genres']
        else:
            movie_title = f"Película ID {request.movie_id}"
            movie_genres = "Desconocido"
        
        # Determinar nivel de gusto
        if prediction >= 4.5:
            taste_level = "Te encantará"
            emoji = "😍"
        elif prediction >= 4.0:
            taste_level = "Te gustará mucho"
            emoji = "😊"
        elif prediction >= 3.5:
            taste_level = "Te gustará"
            emoji = "🙂"
        elif prediction >= 2.5:
            taste_level = "Neutral"
            emoji = "😐"
        else:
            taste_level = "Probablemente no te guste"
            emoji = "😕"
        
        interpretation = (
            f"{emoji} PREDICCIÓN DE RATING\n"
            f"Usuario: {request.user_id}\n"
            f"Película: {movie_title}\n"
            f"Géneros: {movie_genres}\n"
            f"Rating predicho: {prediction:.1f}/5.0\n"
            f"Predicción: {taste_level}\n"
            f"Confianza: {confidence:.1f}%"
        )
        
        return MovieRatingResponse(
            predicted_rating=round(prediction, 2),
            confidence=round(confidence, 1),
            model_info={
                "model_type": model_data['model_info']['type'],
                "user_id": request.user_id,
                "movie_id": request.movie_id,
                "movie_title": movie_title
            },
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
def health():
    try:
        model_data, movies_df, ratings_df = load_movies_model_and_data()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": model_data['model_info']['type'],
            "movies_count": len(movies_df),
            "ratings_count": len(ratings_df)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🎬 Movies Recommendation API")
    uvicorn.run(app, host="0.0.0.0", port=8002)