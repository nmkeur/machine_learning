# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
import pandas as pd
import numpy as np
from classifier import Classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from CreatePlot import CreatePlot
from sklearn.model_selection import StratifiedKFold

class SVM(Classifier):
    def __init__(self,x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.SVC_cls = ""
        self.y_pred = ""
        self.best_params_ = ""
        self.name = "SVM"
        self.setupCV()
        self.grid_search()


    def trainModel(self):
        # Train the model with the best parameters learned in the grid search
        print("Train the model with the best parameters on the full training dataset.")

        #Initialize a Random forest model object
        SVM = SVC(kernel='rbf',probability=True,
                  C= self.best_params_.get('C'),
                  gamma = self.best_params_.get('gamma'),
                  random_state=1,
                  n_jobs=-1)

        #Train the Random forest model object
        SVM.fit(self.x_train,self.y_train)
        #Predict the y values using the Random forest model object
        self.y_pred = SVM.predict(self.x_test)

        print("Detailed classification report:")
        print()
        print("The model is trained with new paramters on the full training dataset.")
        print("The scores are computed on the full test  dataset.")
        print()
        print(classification_report(self.y_test, self.y_pred))

        CP = Createplot(self.name)
        # Plot the results.
        CP.plot_confusion_matrix(self.y_test , self.y_pred, save=True)
        CP.plot_roc_curve(SVM, self.x_test , self.y_test,save=True)
        CP.plot_precision_recall_curve(SVM, self.x_test , self.y_test, save=True)

        #self.scoreModel()

    def grid_search(self):
        CP = CreatePlot(self.name)
        scores = ['f1']
        #cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=i)
        #np.logspace(-10,10,num=21,base=10)
        #[1e-05,1e-04,1e-03,1e-02,0.1,1,10,100,1000]
        #np.logspace(-12,3,num=16,base=10 has good results
        parameters = {'C': [25,50,100,200,250],
        'gamma': np.logspace(-6,-5,num=10,base=10)}
        parameters2 = {'C': [25,50],
        'gamma': np.logspace(-6,-5,num=4,base=10)}
        print("Perform gridsearch with the following options:", parameters)
        SVM = SVC(kernel='rbf',random_state=1,probability=True)
        for score in scores:
            SVC_cls = GridSearchCV(SVM,
                        parameters2, cv=self.skfold,
                         verbose=1,n_jobs=-1, scoring='f1')

            SVC_cls.fit(self.x_train,self.y_train)

            print("Best parameters found on the training dataset:")
            print("Best value for C found: ", SVC_cls.best_params_.get('C'))
            print("Number value for Gamma found: ", SVC_cls.best_params_.get('gamma'))

            y_pred = SVC_cls.predict(self.x_test)

            print("# Tuning hyper-parameters for %s" % score)
            print("Best parameters set found on development set:\n")
            print(SVC_cls.best_params_)
            #Save the best paramters found during the grid search.
            self.best_params_ = SVC_cls.best_params_

            print("Grid scores on the train dataset:\n")
            means = SVC_cls.cv_results_['mean_test_score']
            stds = SVC_cls.cv_results_['std_test_score']


            for mean, std, params in zip(means, stds, SVC_cls.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


            #Create a plot with the gridsearch results.
            CP.plot_grid_search(SVC_cls.cv_results_, parameters.get('gamma'),
                    parameters.get('C'), 'gamma', 'C', True)
        self.trainModel()
        #self.trainModel()
#svclassifierlinear = SVC(kernel='linear')
#svclassifierlinear.fit(x_train, y_train)

#svclassifiernonlinear = SVC(kernel='poly', degree=8)
#svclassifiernonlinear.fit(x_train, y_train)

#svclassifierrbf = SVC(kernel='rbf')
#svclassifierrbf.fit(x_train, y_train)

#svclassifiersigmoid = SVC(kernel='sigmoid',probability=True)
#svclassifiersigmoid.fit(x_train, y_train)

#y_pred = svclassifierlinear.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifiernonlinear.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifierrbf.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#y_pred = svclassifiersigmoid.predict(x_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
