# Loosely following from https://www.kaggle.com/code/dansbecker/basic-data-exploration/tutorial

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# dataset from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_file_path = '../dataset/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna()

print(melbourne_data.describe())

print(melbourne_data.columns)

y = melbourne_data['Price']
feature_names = melbourne_data[['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']]
X = feature_names

melbourne_model = DecisionTreeRegressor(random_state=0)
melbourne_model.fit(X, y)

predictions = melbourne_model.predict(X)

print(melbourne_data.head())
print(predictions[:10])