import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import os


EXTRACT_DIR = "ml-latest-small"

ratings = pd.read_csv(os.path.join(EXTRACT_DIR, "ratings.csv")) 
movies_with_tags_features = pd.read_csv("movies_with_tags_features.csv") 
data = pd.merge(ratings, movies_with_tags_features, on="movieId")

X = data.drop(columns=["userId", "movieId", "title", "rating"])
y = data["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

dump(model, "models_bin/random_forest.joblib")