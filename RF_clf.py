import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from classifier import Classifier
from CreatePlot import CreatePlot
from sklearn import metrics
from classifier import Classifier
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

class RFclass(Classifier):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.RF_clf = ""
        self.y_pred = ""
        self.skfold = ""
        self.best_params_ = ""
        self.setupCV()
        self.gridSearch()
        #self.trainModel()

    def trainModel(self):
        print("Train the new model with new paramters.")
        rf = RandomForestClassifier(oob_score = True,
                                    n_estimators = self.best_params_.get('n_estimators'),
                                    max_features = self.best_params_.get('max_features'),
                                    random_state=1,
                                    n_jobs=-1)
        rf.fit(self.x_train,self.y_train)
        self.y_pred = rf.predict(self.x_test)
        CreatePlot().plot_confusion_matrix(self.y_test , self.y_pred)
        CreatePlot().plot_roc_curve(rf, self.x_test , self.y_test)
        CreatePlot().plot_precision_recall_curve(rf, self.x_test , self.y_test)


        self.scoreModel(rf)
        #self.predictModel()
        #self.gridSearch()

    def featureSelection():
        # Feature importance is a standard metric in RF and can be used for feature selection
        # Here is select the top 10
        feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
        feature_imp[:10]

    def scoreModel(self,rf):
        print("Score the model")
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        scores = cross_validate(rf, self.x_test,
                                self.y_pred,cv=self.skfold,
                                return_train_score=False)
        print(scores)
        self.saveModel(rf)


    def gridSearch(self):
        parameters = {'n_estimators':[250,500],# [500,1000,1500,2000]
                      'max_features':[0.025, 0.05, 0.075, 0.1, 0.125,
                                      0.15, 0.175,0.2, 0.225,0.25]}

        print("Perform gridsearch with the following options:", parameters)
        scores = ['f1']# 'precision','recall'

        rf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs=-1)
        for score in scores:
            clf = GridSearchCV(rf, parameters, cv=self.skfold,
                                        verbose=1, n_jobs=-1, scoring= score)
            clf.fit(self.x_train,self.y_train)

            print("Best parameters found on the training dataset using ",score," as metric:")
            print("Number of estimators: ", clf.best_params_.get('n_estimators'))
            print("Number of maximum features: ", clf.best_params_.get('max_features'))
            y_pred = clf.predict(self.x_test)
            print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
            print("Classification report for the test dataset:")
            print(classification_report(self.y_test, y_pred))

            CreatePlot().plot_grid_search(clf.cv_results_, parameters.get('n_estimators'),
                            parameters.get('max_features'), 'N Estimators', 'Max Features', False)

            print("# Tuning hyper-parameters for %s" % score)
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            self.best_params_ = clf.best_params_
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            bindex = clf.best_index_
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                #print( bindex)
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            #y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(self.y_test, y_pred))
            print()
            #print(RF_clf)
            #CP.plot_confusion_matrix(self.y_test , self.y_pred)
            #CP.plot_precision_recall_curve(RF_clf, self.x_test , self.y_test)
            #CP.plot_roc_curve(RF_clf, self.x_test , self.y_test)
            #self.RF_clf = clf
        # Create new classsifier, this time with optimal parameters.

        #print(self.x_train)

        #feature_imp = pd.Series(rf.feature_importances_,
        #                        index=self.x_train.columns).sort_values(ascending=False)
        #feature_imp.index.name == "index"
        #feature_list = feature_imp.index.tolist()

        #print(feature_list[:50])

        #self.Run2(feature_list[:50])
        self.trainModel()



    def Run2(self,feature_list):
        x_train = self.x_train[feature_list]
        x_test = self.x_test[feature_list]
        #x_train = self.x_train[,feature_list]
        #print(x_train)
        parameters = {'n_estimators':[100,200,300,400,500],# [500,1000,1500,2000]
                      'max_features':[5,10,15,20,25,30,35,40]}
        scores = ['f1']# 'precision','recall'

        CP = CreatePlot()
        rf = RandomForestClassifier(oob_score=True,random_state=1,n_jobs=-1)
        for score in scores:
            clf = GridSearchCV(rf, parameters, cv=self.skfold,
                                        verbose=1, n_jobs=-1,
                                        scoring= score)
            clf.fit(x_train,self.y_train)

            print("Best parameters found on the training dataset using ",score," as metric:")
            print("Number of estimators: ", clf.best_params_.get('n_estimators'))
            print("Number of maximum features: ", clf.best_params_.get('max_features'))

            y_pred = clf.predict(x_test)
            print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
            print("Classification report for the test dataset:")
            print(classification_report(self.y_test, y_pred))


            print("# Tuning hyper-parameters for %s" % score)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            bindex = clf.best_index_
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                #print( bindex)
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            #y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(self.y_test, y_pred))
            print()


    def saveModel(self, rf):
        # save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(rf, open(filename, 'wb'))
            #print(RF_clf)
            #CP.plot_confusion_matrix(self.y_test , self.y_pred)
            #CP.plot_precision_recall_curve(RF_clf, self.x_test , self.y_test)
            #CP.plot_roc_curve(RF_clf, self.x_test , self.y_test)
            #export_csv = (feature_imp[:250].index).to_csv ('export_dataframe.csv', header=False,) #Don't forget to add '.csv' at the end of the path


        #for name, importance in zip(self.x_test.index(), rf.feature_importances_):
        #    print(name, "=", importance)
        #scores = cross_validate(rf , self.x_test, self.y_test, cv=self.skfold, scoring='f1',
                                #return_estimator=True ,return_train_score=False)
        #print(scores)

        #CP.plot_confusion_matrix(self.y_test , ypred)
        #P.plot_precision_recall_curve(rf, self.x_test , self.y_test)
        #CP.plot_roc_curve(rf, self.x_test , self.y_test)

        #CP.plot_learning_curve(rf, self.x_train , self.y_train)
        # Calling Method

                            #cv=self.cv, return_train_score=True)
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
