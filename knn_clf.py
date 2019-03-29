from classifier import Classifier
from sklearn import *
import pandas as pd
from sklearn import metrics
from CreatePlot import CreatePlot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle


class KNN_clf(Classifier):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.RF_clf = ""
        self.y_pred = ""
        self.skfold = ""
        self.best_params_ = ""
        self.name = "KNN"
        self.setupCV()
        self.gridSearch()
        #self.gridSearch()
        #self.trainModel()

    def trainModel(self):
        # Train the model with the best parameters learned in the grid search
        print("Train the model with the best parameters on the full training dataset.")

        #Initialize a Random forest model object
        knn_model = KNeighborsClassifier(n_neighbors= self.best_params_.get('n_neighbors'),
                  p = self.best_params_.get('p'))

        #Train the Random forest model object
        knn_model.fit(self.x_train,self.y_train)
        #Predict the y values using the Random forest model object
        self.y_pred = knn_model.predict(self.x_test)

        print("Detailed classification report:")
        print()
        print("The model is trained with new paramters on the full training dataset.")
        print("The scores are computed on the full test  dataset.")
        print()
        print(classification_report(self.y_test, self.y_pred))

        CP = CreatePlot(self.name)
        # Plot the results.
        CP.plot_confusion_matrix(self.y_test , self.y_pred, save=True)
        CP.plot_roc_curve(knn_model, self.x_test , self.y_test,save=True)
        CP.plot_precision_recall_curve(knn_model, self.x_test , self.y_test, save=True)
        self.saveModel(knn_model)
        #self.scoreModel()


    def gridSearch(self):
        CP = CreatePlot(self.name)
        # construct the set of hyperparameters to tune
        parameters = {"n_neighbors": np.arange(4, 14, 1),
	       "p": [1,2,3,4]}
        scores = ['f1']
                # instantiate learning model (k = 3)
        knn_model = KNeighborsClassifier()
        for score in scores:
            clf = GridSearchCV(knn_model, parameters, cv=self.skfold,
                                        verbose=1, n_jobs=-1,
                                        scoring= score)
            clf.fit(self.x_train,self.y_train)

            print("Best parameters found on the training dataset using ",score," as metric:")
            print("Number of neigbours: ", clf.best_params_.get('n_neighbours'))
            print("Metric used: ", clf.best_params_.get('p'))

            y_pred = clf.predict(self.x_test)


            CP.plot_grid_search(clf.cv_results_, parameters.get('n_neighbors'),
                    parameters.get('p'), 'n_neighbors', 'P', False, True)

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
            #bindex = clf.best_index_
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            self.trainModel()

    def saveModel(self, rf):
        # save the model to disk
        filename = 'KNN/finalized_model.sav'
        pickle.dump(rf, open(filename, 'wb'))
        # Predictions/probs on the test dataset
