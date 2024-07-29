import numpy as np 
import pandas as pd
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv('./data/tmdb_5000_movies.csv')
credits = pd.read_csv('./data/tmdb_5000_credits.csv')

# Merge datasets on title
movies = movies.merge(credits, on='title')

# Keeping important columns for recommendation
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Function to convert JSON-like strings to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

# Apply conversion to genres and keywords
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to convert cast JSON-like strings to list with top 3 cast members
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Function to fetch director from crew JSON-like strings
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview from string to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Function to remove spaces from list elements
def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Concatenate all lists into a new 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with the required columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert lists in 'tags' column to strings
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert 'tags' column to lower case
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming the 'tags' column
ps = PorterStemmer()
def stems(text):
    T = []
    for i in text.split():
        T.append(ps.stem(i))
    return " ".join(T)

new_df['tags'] = new_df['tags'].apply(stems)

# Convert tags to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Function to recommend movies
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

# Test recommendation
recommend('Spider-Man 2')

# Save the DataFrame and similarity matrix
pickle.dump(new_df, open('artifacts/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
