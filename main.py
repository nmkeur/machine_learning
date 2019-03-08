import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from SVM_clf import SVMclass

#Path to the file
class ML_TOOL():
    def __init__(self):
        self.datafile_path = '~/Documents/machine_learning/counts_norm_cleaned.csv'
        #Global variables, makes it easyier to debug.
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""
        self.readFile()

    def readFile(self):
        #Read the data and store data in pandas DataFrame
        norm_data = pd.read_csv(self.datafile_path)
        y_value = norm_data.patientgroup
        #x_value = norm_data[feature_names]

        # Select all column from the dataframe except the first (sampleID) and last (patientgroup)
        x_value = norm_data[norm_data.columns[1:12490]]
        # Split the data in train and test set.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_value, y_value, test_size=0.3)
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)
        print(rskf)
        print("Train dataset has {} samples and {} attributes".format(*self.x_train.shape))
        print("Test dataset has {} samples and {} attributes".format(*self.x_test.shape))


def startRF(args):
        print("StartRF")
        print(t.k)

def startSVM(args):
    #SVMclass(args.k,args.cv)
    SVM = SVMclass(args.k)

def main():
    ML = ML_TOOL()
    #Create top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    #Create parser for SVM
    parser_rf = subparsers.add_parser('rf')
    parser_rf.set_defaults(func=startRF)
    parser_foo = subparsers.add_parser('svm')
    parser_foo.add_argument('-k',
                    required=True,
                    type=str,
                    choices=["linear","rbf"],
                    default="linear")
    parser_foo.set_defaults(func=startSVM)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
