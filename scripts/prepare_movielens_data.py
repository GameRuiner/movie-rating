import os
import zipfile
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, concat_ws

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_FILE = "ml-latest-small.zip"
EXTRACT_DIR = "ml-latest-small"
spark = SparkSession.builder.appName("MovieLensPreprocessing").getOrCreate()

if not os.path.exists(ZIP_FILE):
    print("🔽 Pobieranie danych MovieLens...")
    response = requests.get(URL)
    with open(ZIP_FILE, "wb") as f:
        f.write(response.content)
    print("✅ Plik ZIP pobrany.")

if not os.path.exists(EXTRACT_DIR):
    print("📦 Rozpakowywanie...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall()
    print("✅ Rozpakowano do katalogu:", EXTRACT_DIR)

print("📚 Wczytywanie danych CSV...")
movies_spark = spark.read.csv(os.path.join(EXTRACT_DIR, "movies.csv"), header=True, inferSchema=True)
movies = movies_spark.toPandas()
tags_spark = spark.read.csv(os.path.join(EXTRACT_DIR, "tags.csv"), header=True, inferSchema=True)

print("🎭 Przetwarzanie gatunków...")
genres = movies["genres"].str.get_dummies(sep="|")
movies = pd.concat([movies[["movieId", "title"]], genres], axis=1)

print("🔖 Łączenie tagów...")
tags_spark_grouped = tags_spark.groupBy("movieId").agg(concat_ws(" ", collect_list("tag")).alias("tag"))
tags_grouped = tags_spark_grouped.toPandas()
movies = pd.merge(movies, tags_grouped, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")

print("🔠 Tworzenie TF-IDF z tagów...")
tfidf = TfidfVectorizer(max_features=300, stop_words="english")
tag_features = tfidf.fit_transform(movies["tag"]).toarray()
tag_df = pd.DataFrame(tag_features, columns=[f"tag_{w}" for w in tfidf.get_feature_names_out()])

print("🧩 Tworzenie końcowego zbioru danych...")
final_df = pd.concat([movies.drop(columns=["tag"]), tag_df], axis=1)

final_df.to_csv("movies_with_tags_features.csv", index=False)
print("✅ Gotowe! Plik zapisany jako 'movies_with_tags_features.csv'")