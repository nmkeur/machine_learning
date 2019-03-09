import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.metrics import recall_score
#from sklearn.metrics import mean_absolute_error
from sklearn import metrics
#import matplotlib.pyplot as plt
#import seaborn as sns

class RFclass():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.RF_clf = ""
        self.cv = 5
        self.y_pred = ""
        self.skfold = ""
        self.setupCV()
        self.trainModel()

    def trainModel(self):
        self.RF_clf = RandomForestClassifier(n_estimators=5000,oob_score=True,random_state=1)
        self.RF_clf.fit(self.x_train,self.y_train)
        self.y_pred = self.RF_clf.predict(self.x_test)
        self.scoreModel()
        #self.predictModel()

    def featureSelection():
        # Feature importance is a standard metric in RF and can be used for feature selection
        # Here is select the top 10
        feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
        feature_imp[:10]

    def scoreModel(self):
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        self.RF_clf = RandomForestClassifier(n_estimators=5000,oob_score=True,random_state=1)
        scores = cross_validate(self.RF_clf, self.x_test, self.y_pred,cv=self.skfold, return_train_score=False)
        print(scores)

    def setupCV(self):
        n_folds=5
        n_repeats=10
        self.skfold = RepeatedStratifiedKFold(n_splits=n_folds,n_repeats=n_repeats,
                                         random_state=0)
    def predictModel(self):
        pass
        #y_pred = self.RF_clf.predict(self.x_test)

        #print(scores)
        #print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
        #scores = cross_validate(self.RF_clf, self.x_test, y_pred,cv=self.cv, return_train_score=False)
        #print(scores)
# Create the list of features below
#feature_names = ["ENSG00000125868","ENSG00000143198","ENSG00000124275","ENSG00000162521","ENSG00000178952","ENSG00000082212","ENSG00000204272","ENSG00000166428","ENSG00000169756","ENSG00000105369"]
# select data corresponding to features in feature_names




#Train the model using the training sets y_pred=clf.predict(X_test)


#Predict the Y variable on the test dataset using the trained model.
#Show the scores with cross validation set to  5




# Creates a plot for with the feature importances.
#%matplotlib inline
# Creating a bar plot
#sns.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
# Add labels to your graph
#plt.xlabel('Feature Importance Score')
#plt.ylabel('Features')
#plt.title("Visualizing Important Features")
#plt.show()
