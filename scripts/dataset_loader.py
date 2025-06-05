import os
import pandas as pd
from sklearn.model_selection import train_test_split

class MovieRatingsDataset:
    def __init__(self, extract_dir="../ml-latest-small", tag_features_path="../movies_with_tags_features.csv", test_size=0.2, random_state=42):
        self.extract_dir = extract_dir
        self.tag_features_path = tag_features_path
        self.test_size = test_size
        self.random_state = random_state

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        ratings = pd.read_csv(os.path.join(self.extract_dir, "ratings.csv"))
        movie_features = pd.read_csv(self.tag_features_path)
        self.data = pd.merge(ratings, movie_features, on="movieId")
        X = self.data.drop(columns=["userId", "movieId", "title", "rating", "timestamp"])
        y = self.data["rating"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def get_train_test(self):
        if self.X_train is None:
            self.load_data()
        return self.X_train, self.X_test, self.y_train, self.y_test