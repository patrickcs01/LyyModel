# https://www.tensorflow.org/datasets/overview
# import tensorflow_datasets as tfds
# import ssl
# import os
#
# os.environ['NO_GCE_CHECK'] = 'true'
#
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'https://127.0.0.1:7890'
# ssl._create_default_https_context = ssl._create_unverified_context
#
# ratings = tfds.load('movielens/100k-ratings', split="train")
# movies = tfds.load('movielens/100k-movies', split="train")
# movies = movies.map(lambda x: x["movie_title"])

import pandas as pd

df = pd.read_csv('../dataset/movies.csv')
column = ['index', 'budget', 'genres', 'homepage', 'id', 'keywords',
       'original_language', 'original_title', 'overview', 'popularity',
       'production_companies', 'production_countries', 'release_date',
       'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title',
       'vote_average', 'vote_count', 'cast', 'crew', 'director']

print(df)






