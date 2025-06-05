from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from dataset_loader import MovieRatingsDataset


dataset = MovieRatingsDataset()
X_train, X_test, y_train, y_test = dataset.get_train_test()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

dump(model, "models_bin/random_forest_no_timestamp.joblib")