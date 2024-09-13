import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import train_test_split as tts

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
mae = tts.mean_absolute_error(y, predicted_home_prices)
print('\nMean Absolute Error (MAE): ', mae)