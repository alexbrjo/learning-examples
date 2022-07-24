# Loosely following from https://www.kaggle.com/code/dansbecker/basic-data-exploration/tutorial

# Suggested features
# MAPE = 0.23646279738719264 for 
# Features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
#
# All Numeric features
# MAPE = 0.0013906121511715005 for
# Features = ['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
#      'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# dataset from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_file_path = '../dataset/melb_data.csv'

# load and clean (drop na and non-numeric columns lol) data
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna()

y = melbourne_data['Price']
X = melbourne_data[['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']]

# train
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor(random_state=0)
melbourne_model.fit(train_X, train_y)

# predict
predictions = melbourne_model.predict(val_X)

print("Features: ")
print(X.columns)
print("DecisionTreeRegressor MAPE: " + str(mean_absolute_percentage_error(val_y, predictions)))