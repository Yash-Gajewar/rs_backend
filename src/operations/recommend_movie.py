import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def recommend_movie(movie_name):
    # Load the data
    movies_df = pd.read_csv('https://portfolio-project-images-yash.s3.ap-south-1.amazonaws.com/movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
    rating_df = pd.read_csv('https://portfolio-project-images-yash.s3.ap-south-1.amazonaws.com/ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    
    # Merge data
    df = pd.merge(rating_df, movies_df, on='movieId')
    combine_movie_rating = df.dropna(axis=0, subset=['title'])
    
    # Count movie ratings
    movie_ratingCount = (combine_movie_rating
                         .groupby(by=['title'])['rating']
                         .count()
                         .reset_index()
                         .rename(columns={'rating': 'totalRatingCount'})
                         [['title', 'totalRatingCount']])
    
    # Merge with rating count
    rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how='left')
    
    # Filter popular movies
    popularity_threshold = 50
    rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    
    # Pivot data to create movie features
    movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    movie_features_df_matrix = csr_matrix(movie_features_df.values)
    
    # Train the model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movie_features_df_matrix)
    
    # Check if the movie exists in the dataset
    if movie_name not in movie_features_df.index:
        return {'error': 'Movie not found'}
    
    # Get recommendations
    query_index = movie_features_df.index.get_loc(movie_name)
    distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendations.append(movie_features_df.index[indices.flatten()[i]])
    
    return recommendations
