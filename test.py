import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# save filepath to variable for easier access
melbourne_file_path = '~/Downloads/train.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
#melbourne_data.describe()
#melbourne_data.columns

#Remove NA Values
#home_data = melbourne_data.dropna(axis=0)

#Value we want to predict
y = melbourne_data.SalePrice

# Create the list of features below
feature_names = ['LotArea',"YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]

# select data corresponding to features in feature_names
X = melbourne_data[feature_names]

#print(X.describe())
#X.head()

home_model = DecisionTreeRegressor(random_state=1)

# Fit the model
home_model.fit(X,y)
predictions = home_model.predict(X)
print(predictions)


predicted_home_prices = home_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# Split data in train and test set

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


print("First in-sample predictions:", melbourne_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

    # compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5,25,50,100,250,500,1000,2500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

final_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=1)
final_model.fit(X, y)
