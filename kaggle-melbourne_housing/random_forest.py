# Loosely following from https://www.kaggle.com/code/dansbecker/basic-data-exploration/tutorial

# Best decision tree
# max_leaf_nodes: 1400 Features: 13 DecisionTreeRegressor MAPE: 0.0014673169224195199
#


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# dataset from https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
melbourne_file_path = '../dataset/melb_data.csv'

# load and clean (drop na and non-numeric columns lol) data
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
melbourne_data = melbourne_data.dropna()

y = melbourne_data['Price']
X = melbourne_data[['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']]

# train
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# predict
def get_mae(train_X, val_X, train_y, val_y):
    melbourne_model = RandomForestRegressor(random_state=0)
    melbourne_model.fit(train_X, train_y)
    predictions = melbourne_model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    print("Features: " + str(len(X.columns)) + 
        " DecisionTreeRegressor MAPE: " + str(mean_absolute_percentage_error(val_y, predictions)))
    return(mae)

get_mae(train_X, val_X, train_y, val_y)