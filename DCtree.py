import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
#from sklearn.metrics import recall_score
#from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Path to the file
datafile_path = '~/Documents/machine_learning/counts_norm_cleaned.csv'

#Read the data and store data in pandas DataFrame
norm_data = pd.read_csv(datafile_path)

#Here we print a summary of the data and we print all the the column names in the dataframe.
#norm_data.describe()
#norm_data.columns

#This will be the value we want to predict. (patientgroup)
y_value = norm_data.patientgroup

# Create the list of features below
#feature_names = ["ENSG00000125868","ENSG00000143198","ENSG00000124275","ENSG00000162521","ENSG00000178952","ENSG00000082212","ENSG00000204272","ENSG00000166428","ENSG00000169756","ENSG00000105369"]
# select data corresponding to features in feature_names
#x_value = norm_data[feature_names]

# Select all column from the dataframe except the first (sampleID) and last (patientgroup)
x_value = norm_data[norm_data.columns[1:12490]]

# Split the data in train and test set.
x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.4)

gkf = GroupKFold(n_splits=3)

#Create a simple gaussian Classifier
RF_clf=RandomForestClassifier(n_estimators=5000,oob_score=True,random_state=1)

#Train the model using the training sets y_pred=clf.predict(X_test)
RF_clf.fit(x_train,y_train)

#Predict the Y variable on the test dataset using the trained model.
#Show the scores with cross validation set to  5

y_pred = RF_clf.predict(x_test)
scores = cross_val_score(RF_clf, x_test, y_pred, cv=5)
scores
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

scores = cross_validate(RF_clf, x_test, y_pred,cv=5, return_train_score=False)
scores

# Feature importance is a standard metric in RF and can be used for feature selection
# Here is select the top 10
feature_imp = pd.Series(RF_clf.feature_importances_,index=x_value.columns).sort_values(ascending=False)
feature_imp[:10]

'''
ENSG00000125868    0.012854
ENSG00000143198    0.012550
ENSG00000124275    0.010922
ENSG00000162521    0.009562
ENSG00000178952    0.008755
ENSG00000082212    0.007788
ENSG00000204272    0.007495
ENSG00000166428    0.007425
ENSG00000169756    0.007259
ENSG00000105369    0.007016
dtype: float64
'''

# Creates a plot for with the feature importances.
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
