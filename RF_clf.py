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

#In the case you get DeprecationWarnings you can turn those off
#warnings.filterwarnings("ignore", category=DeprecationWarning)

class RFclass(Classifier):
    #Initialize the global variables.
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.RF_clf = ""
        self.y_pred = ""
        self.skfold = ""
        self.best_params_ = ""
        self.name = "RF"
        self.setupCV()
        self.gridSearch()
        #self.trainModel()

    def trainModel(self):
        # Train the model with the best parameters learned in the grid search
        print("Train the model with the best parameters on the full training dataset.")

        #Initialize a Random forest model object
        rf = RandomForestClassifier(oob_score = True,
                                    n_estimators = self.best_params_.get('n_estimators'),
                                    max_features = self.best_params_.get('max_features'),
                                    random_state=1,
                                    n_jobs=10)
        #Train the Random forest model object
        rf.fit(self.x_train,self.y_train)
        #Predict the y values using the Random forest model object
        self.y_pred = rf.predict(self.x_test)

        print("Detailed classification report:")
        print()
        print("The model is trained with new paramters on the full training dataset.")
        print("The scores are computed on the full test  dataset.")
        print()
        print(classification_report(self.y_test, self.y_pred))

        # Plot the results.
        CP = CreatePlot(self.name)
        CP.plot_confusion_matrix(self.y_test , self.y_pred, save=True)
        CP.plot_roc_curve(rf, self.x_test , self.y_test, save=True)
        CP.plot_precision_recall_curve(rf, self.x_test , self.y_test, save=True)

        self.scoreModel(rf)

    def featureSelection():
        # Feature importance is a standard metric in RF and can be used for feature selection
        # Here is select the top 10
        feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
        feature_imp[:10]

    def scoreModel(self,rf):
        print("Scoring the model")
        print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))
        scores = cross_validate(rf, self.x_test,
                                self.y_pred,cv=self.skfold,
                                return_train_score=False)
        print(scores)
        self.saveModel(rf)


    def gridSearch(self):
        # Start the gridsearch with the following parameters.
        parameters = {'n_estimators':[100,200],# [500,1000,1500,2000]
                      'max_features':[10,20,30]}

        print("Perform gridsearch with the following options:", parameters)
        #Can run the the gridsearch metric specific.
        #For now only run this for the f1 metric
        scores = ['f1']# 'precision','recall'
        #Initialize Random forest object
        rf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs=10, max_depth=100)
        for score in scores:
            #Setup gridsearch
            clf = GridSearchCV(rf, parameters, cv=self.skfold,
                                        verbose=1, n_jobs=-10, scoring= score)
            clf.fit(self.x_train,self.y_train)

            print("Best parameters found on the training dataset using ",score," as metric:")
            print("Number of estimators: ", clf.best_params_.get('n_estimators'))
            print("Number of maximum features: ", clf.best_params_.get('max_features'))

            y_pred = clf.predict(self.x_test)

            print("# Tuning hyper-parameters for %s" % score)
            print("Best parameters set found on development set:\n")
            print(clf.best_params_)
            #Save the best paramters found during the grid search.
            self.best_params_ = clf.best_params_

            print("Grid scores on the train dataset:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            bindex = clf.best_index_
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


            #Create a plot with the gridsearch results.
            CP = CreatePlot(self.name)
            CP.plot_grid_search(clf.cv_results_, parameters.get('n_estimators'),
                    parameters.get('max_features'), 'N Estimators', 'Max Features', False, True)

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
        rf = RandomForestClassifier(oob_score=True,random_state=1,n_jobs=10)
        for score in scores:
            clf = GridSearchCV(rf, parameters, cv=self.skfold,
                                        verbose=1, n_jobs=10,
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
        filename = 'RF/finalized_model.sav'
        pickle.dump(rf, open(filename, 'wb'))
            #print(RF_clf)
            #CP.plot_confusion_matrix(self.y_test , self.y_pred)
            #CP.plot_precision_recall_curve(RF_clf, self.x_test , self.y_test)
            #CP.plot_roc_curve(RF_clf, self.x_test , self.y_test)
            #export_csv = (feature_imp[:250].index).to_csv ('export_dataframe.csv', header=False,) #Don't forget to add '.csv' at the end of the path
            
