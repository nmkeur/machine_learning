import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scikitplot as skplt


class CreatePlot():
    def __init__ (self):
        pass

    def plot_PCA(self, pca_object):
        skplt.decomposition.plot_pca_component_variance(pca)
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
        plt.show()

    def plot_precision_recall_curve(self, rf_object, x_test, y_test):
        pp = rf_object.predict_proba(x_test)
        skplt.metrics.plot_precision_recall_curve(y_test, pp)
        plt.show()

    def plot_roc_curve(self, rf_object, x_test, y_test):
        pp = rf_object.predict_proba(x_test)
        skplt.metrics.plot_roc_curve(y_test, pp)
        plt.show()

    def plot_learning_curve(self, rf_object, x_test, y_test):
        skplt.estimators.plot_learning_curve(rf_object, x_test, y_test)
        plt.show()

    def plot_grid_search(self, cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, log):
        # Get Test Scores Mean and std for each grid searchs

        # Based on https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
        scores_mean = cv_results['mean_test_score']
        scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
        #print(scores_mean)

        scores_sd = cv_results['std_test_score']
        scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))
        #print(scores_sd)

        # Plot Grid search scores
        _, ax = plt.subplots(1,1)

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        for idx, val in enumerate(grid_param_2):
            #ax.fill_between(grid_param_1, scores_mean[idx,:] - scores_sd[idx,:],
            #        scores_mean[idx,:] + scores_sd[idx,:], alpha=0.1)
            ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

        ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
        ax.set_xlabel(name_param_1, fontsize=16)
        ax.set_ylabel('CV Average Score', fontsize=16)
        ax.legend(loc="best", fontsize=15)
        ax.grid('on')
        if log == True:
            ax.set_xscale('log')
        plt.show()
