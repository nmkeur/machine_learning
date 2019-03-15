import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from RF_clf import RFclass
from SVM_linear import SVMlinear
from SVM_rbf import SVMrbf
from SVM_poly import SVMpoly
from SVM_sigmoid import SVMsigmoid
from knn_clf import KNN_clf
from sklearn.preprocessing import StandardScaler

class ML_TOOL():
    def __init__(self):
        #self.datafile_path = '~/Documents/machine_learning/counts_norm_cleaned.csv' laptop
        self.datafile_path = 'norm_reads.csv'

        #Global variables, makes it easyier to debug.
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

        self.readFile()

    def readFile(self):
        #Read the data and store data in pandas DataFrame
        norm_data = pd.read_csv(self.datafile_path, index_col=0)
        y_value = norm_data.patientgroup

        # Select all column from the dataframe except
        # the first (sampleID) and last (patientgroup)
        x_value = norm_data.loc[:, norm_data.columns != 'patientgroup']
        #x_value.loc[:, norm_data.columns != 'patientgroup']
        print(x_value.head())
        # Split and shuffles the data in train and test
        # set in a stratified manner.
        # 70% for training and the remaining to test/validate.
        x_train, x_test, self.y_train, self.y_test = train_test_split(x_value, y_value,
                                        test_size=0.2, shuffle=True, stratify=y_value)
        scaler = StandardScaler()
        scaler.fit(x_train)
        #df_scaled = pd.DataFrame(scaler.fit_transform(x_value),
        #                         index=norm_data.index,
        #                          columns = norm_data.columns)
        # Apply transform to both the training set and the test set.
        self.x_train = pd.DataFrame(scaler.transform(x_train),
                                    index=x_train.index,
                                    columns = x_train.columns)
        self.x_test = pd.DataFrame(scaler.transform(x_test),
                                    index=x_test.index,
                                    columns = x_test.columns)
        #self.x_test = scaler.transform(x_test)
        # Print dataset statistics.
        print("Train dataset has {} samples and {} attributes".format(*self.x_train.shape))
        print("Test dataset has {} samples and {} attributes".format(*self.x_test.shape))



    def startRF(self,args):
        # Function starts the randomforest classifier.
        print("Start random forest algorithm")
        RandomForestClassifier = RFclass(self.x_train, self.y_train, self.x_test, self.y_test)

    def startSVM(self,args):
        if args.k == "linear":
            print("Kernell is linear")
            svm = SVMlinear(self.x_train, self.y_train, self.x_test, self.y_test, args.k)
        elif args.k == "rbf":
            svm = SVMrbf(self.x_train, self.y_train, self.x_test, self.y_test, args.k)
        elif args.k == "poly":
            print("kernell is poly")
            svm = SVMpoly(self.x_train, self.y_train, self.x_test, self.y_test, args.k)
        elif args.k == "sigmoid":
            print("kernell is sigmoid")
            svm = SVMsigmoid(self.x_train, self.y_train, self.x_test, self.y_test, args.k)
        else:
            print("Unknown kernell")
            exit(0)
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
    parser_foo.add_argument('-k',
                    required=True,
                    type=str,
                    choices=["linear","rbf","poly","sigmoid"],
                    default="linear")
    parser_foo.set_defaults(func=ML.startSVM)

    # Parse the arguments on the commandline.
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
