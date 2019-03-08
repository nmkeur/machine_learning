import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
#from sklearn.metrics import recall_score
#from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

class RFclass():
    def __init__(self,kernell,cv):
        self.x_train = ""
        self.y_train = ""
        self.x_test = ""
        self.y_test = ""
        self.kernell = kernell

def featureSelection():

    # Feature importance is a standard metric in RF and can be used for feature selection
    # Here is select the top 10
    feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
    feature_imp[:10]




# Create the list of features below
feature_names = ["ENSG00000125868","ENSG00000143198","ENSG00000124275","ENSG00000162521","ENSG00000178952","ENSG00000082212","ENSG00000204272","ENSG00000166428","ENSG00000169756","ENSG00000105369"]
# select data corresponding to features in feature_names




#Create a simple gaussian Classifier
RF_clf=RandomForestClassifier(n_estimators=5000,oob_score=True,random_state=1, probability=True)

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

# Creates a plot for with the feature importances.
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
