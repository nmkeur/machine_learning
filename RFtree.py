# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

class rf_classifier:
    

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_file_path = '~/Desktop/kaggle_house/train.csv'
train_data = pd.read_csv(train_file_path)

test_file_path = '~/Desktop/kaggle_house/test.csv'
test_data = pd.read_csv(test_file_path)

train_data.describe()

# Create target object and call it y
y = train_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
test_x = test_data[features]

print(test_x)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
home_model = DecisionTreeRegressor(random_state=1)
# Fit Model
home_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = home_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
home_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
home_model.fit(train_X, train_y)
val_predictions = home_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

rf_test_predictions = rf_model.predict(test_x)
print(len(rf_test_predictions))


output = pd.DataFrame({'Id': test_data.Id,'SalePrice': rf_test_predictions})
output.to_csv('submission.csv', index=False)
