import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
# print(melbourne_data.describe())

# display list of column names
print('Columns:')
print(melbourne_data.columns, '\n')

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# prediction target
print('First few prices:')
y = melbourne_data.Price
print(y.head())

# Choose features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Create subset of dataframe, comprised of only features as columns
X = melbourne_data[melbourne_features]

##############################################
# Decision tree example

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# Predictions
print("\nMaking price predictions for the following 5 houses:")
print(X.head())
print("\nThe predictions are:")
print(melbourne_model.predict(X.head()))

# Mean absolute Error (MAE)
predicted_home_prices = melbourne_model.predict(X)
print('\nMean absolute error:')
print(mean_absolute_error(y, predicted_home_prices))

# Importance of train test split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print('\nTrain test split :')
print(mean_absolute_error(val_y, val_predictions))

#################################################
#overfitting vs underfitting (controlling tree depth)
# OVERFITTING occurrs when the there's too many divisions (leaf_nodes) within a model's calculation process.
# The prediction would be an output that's very close to the samples within the training data.
# So when new data is provided the predictions will skew.
# (capturing spurious patterns that won't recur in the future, leading to less accurate predictions)

# UNDERFITTING occurrs when there is not enough divisions within a model's calculation process.
# The prediction would be an output that's no where near correct because the model has not
# undergone enough training.
# NOTE: changing prediction features could prove beneficial
# (failing to capture relevant patterns, again leading to less accurate predictions)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))