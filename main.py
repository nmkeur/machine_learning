import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from RF_clf import RFclass
from SVMclf import SVM
from knn_clf import KNN_clf
from sklearn.preprocessing import StandardScaler

class ML_TOOL():
    def __init__(self):
        #self.datafile_path = '~/Documents/machine_learning/counts_norm_cleaned.csv' lapto
        #Global variables, makes it easyier to debug.
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

        self.readFile()

    def readFile(self):
        #Read the data and store data in pandas DataFrame
        datafile_path_test = 't_normalised_filter_test(0.8_1).csv'
        datafile_path_train = 't_normalised_filter_train(0.8_1).csv'
        test_data = pd.read_csv(datafile_path_test, index_col=0, delimiter=";")
        train_data = pd.read_csv(datafile_path_train, index_col=0, delimiter=";")

        self.y_train = train_data.Type
        self.y_test = test_data.Type
        # Select all column from the dataframe except
        # the first (sampleID) and last (patientgroup)
        x_data_test = test_data.loc[:, test_data.columns != 'Type']
        x_data_train = train_data.loc[:, train_data.columns != 'Type']

        scaler = StandardScaler()
        scaler.fit(x_data_train)
        #df_scaled = pd.DataFrame(scaler.fit_transform(x_value),
        #                         index=norm_data.index,
        #                          columns = norm_data.columns)
        # Apply transform to both the training set and the test set.
        self.x_train = pd.DataFrame(scaler.transform(x_data_train),
                                    index=x_data_train.index,
                                    columns = x_data_train.columns)
        self.x_test = pd.DataFrame(scaler.transform(x_data_test),
                                    index=x_data_test.index,
                                    columns = x_data_test.columns)
        # Print dataset statistics.
        print("Train dataset has {} samples and {} attributes".format(*self.x_train.shape))
        print("Test dataset has {} samples and {} attributes".format(*self.x_test.shape))

    def startRF(self,args):
        # Function starts the randomforest classifier.
        print("Start random forest algorithm")
        RandomForestClassifier = RFclass(self.x_train, self.y_train, self.x_test, self.y_test)

    def startSVM(self,args):
        print("Starting the SVM algorithm")
        svm = SVM(self.x_train, self.y_train, self.x_test, self.y_test)
    def startKNN(self,args):
        print("Start KNN algorithm")
        RandomForestClassifier = KNN_clf(self.x_train, self.y_train, self.x_test, self.y_test)



def main():
    ML = ML_TOOL()
    # Initialize top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # Initialize parser for KNN
    # and call start of the algo.
    parser_knn = subparsers.add_parser('knn')
    parser_knn.set_defaults(func=ML.startKNN)
    # Initialize parser for RF
    # and call start of the algo.
    parser_rf = subparsers.add_parser('rf')
    parser_rf.set_defaults(func=ML.startRF)

    # Initialize parser for SVM and sub arguments
    # and call start of the algo.
    parser_foo = subparsers.add_parser('svm')
    #parser_foo.add_argument('-k',
    #                required=True,
    #                type=str,
    #                choices=["linear","rbf","poly","sigmoid"],
    #                default="linear")
    parser_foo.set_defaults(func=ML.startSVM)

    # Parse the arguments on the commandline.
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
