import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Setup Data (Simulated User-Item Ratings Matrix) ---
# Rows are Users, Columns are Movies. Ratings are 1-5.
# NaN means the user has not rated the movie.
data = {
    'User_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'Movie_Title': [
        'Star Wars', 'The Matrix', 'Inception', 'Star Wars', 'Titanic', 'Avatar',
        'Inception', 'Avatar', 'Pulp Fiction', 'The Matrix', 'Inception', 'Pulp Fiction',
        'Star Wars', 'The Matrix', 'Titanic', 'Avatar'
    ],
    'Rating': [5, 4, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 3, 2]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the User-Item Matrix
# This pivots the data so users are rows and movies are columns, with ratings as values.
user_movie_matrix = df.pivot_table(index='User_ID', columns='Movie_Title', values='Rating')

print("--- User-Item Rating Matrix ---")
print(user_movie_matrix)
print("-" * 35)

# --- 2. Calculate Item Similarity (Cosine Similarity) ---

# Transpose the matrix to have Movies as rows for Item-Based CF
movie_user_matrix = user_movie_matrix.fillna(0).T # Fill NaN with 0 for similarity calculation
# 

# Calculate the Cosine Similarity between movies
# Cosine Similarity measures the cosine of the angle between two non-zero vectors.
# If the vectors are close (similar), the cosine is close to 1.
similarity_matrix = cosine_similarity(movie_user_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_user_matrix.index, columns=movie_user_matrix.index)

print("--- Movie Similarity Matrix (Top-Left Corner) ---")
print(similarity_df.head())
print("-" * 35)


# --- 3. The Recommendation Function ---
def get_recommendations(user_id, similarity_df, user_movie_matrix, num_recommendations=3):
    """
    Suggests top 'N' movies to a specific user.
    """
    
    # 1. Get the target user's ratings
    user_ratings = user_movie_matrix.loc[user_id].dropna()
    
    # 2. Find unrated movies
    all_movies = user_movie_matrix.columns
    rated_movies = user_ratings.index
    unrated_movies = all_movies.drop(rated_movies, errors='ignore')
    
    # Initialize a dictionary to store the predicted score for each unrated movie
    predicted_ratings = {}

    # Iterate through each unrated movie
    for movie in unrated_movies:
        # Get the similarity scores between the current unrated movie and all rated movies
        # We only look at movies the user HAS rated to make a prediction
        similar_to_rated = similarity_df[movie].loc[rated_movies]
        
        # Calculate the weighted average rating (Prediction)
        # Prediction = sum(Rating_of_rated_movie * Similarity_to_unrated_movie) / sum(Absolute_Similarities)
        
        # Element-wise multiplication of the user's ratings and the similarity scores
        numerator = (user_ratings * similar_to_rated).sum()
        
        # Sum of absolute similarity scores for normalization
        denominator = similar_to_rated.abs().sum()
        
        if denominator != 0:
            predicted_ratings[movie] = numerator / denominator
        else:
            predicted_ratings[movie] = 0 # Can't calculate, so set to 0

    # 3. Sort predictions and get the top N
    predicted_series = pd.Series(predicted_ratings)
    
    # Sort in descending order
    recommendations = predicted_series.sort_values(ascending=False)
    
    return recommendations.head(num_recommendations)


# --- 4. Run the System for a Target User ---
target_user_id = 3
print(f"--- Top {3} Recommendations for User {target_user_id} ---")
recommendations = get_recommendations(target_user_id, similarity_df, user_movie_matrix, num_recommendations=3)

# Display the results
if not recommendations.empty:
    print(f"\nUser {target_user_id} previously rated:\n{user_movie_matrix.loc[target_user_id].dropna()}")
    print("\nRecommended Movies:")
    print(recommendations.to_string())
else:
    print("\nNo new recommendations could be generated (perhaps the user rated all available items or has no similar items).")

print("-" * 35)