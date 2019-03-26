from classifier import Classifier
from sklearn import *
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


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
        self.RUNknn()
        #self.gridSearch()
        #self.trainModel()


    def RUNknn(self):
        # construct the set of hyperparameters to tune
        parameters = {"n_neighbors": np.arange(2, 24, 1),
	       "metric": ["euclidean", "cityblock"]}
        scores = ['f1']
                # instantiate learning model (k = 3)
        knn_model = KNeighborsClassifier()
        for score in scores:
            clf = GridSearchCV(knn_model, parameters, cv=self.skfold,
                                        verbose=25, n_jobs=-1,
                                        scoring= score)
            clf.fit(self.x_train,self.y_train)

            print("Best parameters found on the training dataset using ",score," as metric:")
            print("Number of estimators: ", clf.best_params_.get('n_estimators'))
            print("Number of maximum features: ", clf.best_params_.get('max_features'))

            y_pred = clf.predict(self.x_test)
            print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
            print("Classification report for the test dataset:")
            print(classification_report(self.y_test, y_pred))

            #print(self.skfold)


            #CP.plot_grid_search(clf.cv_results_, parameters.get('n_estimators'),
            #                parameters.get('max_features'), 'N Estimators', 'Max Features')
            #print(RF_clf.cv_results_)

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
            #print(RF_clf)
            #CP.plot_confusion_matrix(self.y_test , self.y_pred)
            #CP.plot_precision_recall_curve(RF_clf, self.x_test , self.y_test)
            #CP.plot_roc_curve(RF_clf, self.x_test , self.y_test)
            self.RF_clf = clf
        # Create new classsifier, this time with optimal parameters.
        #rf = KNeighborsClassifier(oob_score = True,
        #                            n_estimators = self.RF_clf.best_params_.get('n_estimators'),
        #                            max_features = self.RF_clf.best_params_.get('max_features'),
        #                            random_state=1,
        #                            n_jobs=-1)
        #rf.fit(self.x_train,self.y_train)
        #ypred = rf.predict(self.x_test)

        # fit the model
        knn_model.fit(self.x_train, self.y_train)
        # Accuracy
        knn_model.score(self.x_train, self.y_train)

        # Predictions/probs on the test dataset
        predicted = pd.DataFrame(knn_model.predict(self.x_test))
        probs = pd.DataFrame(knn_model.predict_proba(self.x_test))

        # Store metrics
        knn_accuracy = metrics.accuracy_score(self.y_test, predicted)
        knn_roc_auc = metrics.roc_auc_score(self.y_test, probs[1])
        knn_confus_matrix = metrics.confusion_matrix(self.y_test, predicted)
        knn_classification_report = metrics.classification_report(self.y_test, predicted)
        knn_precision = metrics.precision_score(self.y_test, predicted, pos_label=1)
        knn_recall = metrics.recall_score(self.y_test, predicted, pos_label=1)
        knn_f1 = metrics.f1_score(self.y_test, predicted, pos_label=1)

        print(knn_classification_report)
        # Evaluate the model using 10-fold cross-validation
        knn_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), self.x_test,
                                        self.y_test, scoring='precision', cv=10)
        knn_cv_mean = np.mean(knn_cv_scores)
