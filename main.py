import pandas as pd
import Data_exploring as eda
from sklearn.preprocessing import StandardScaler


def printHelp():
    print("#### HELP ####")
    print("H -> printing HELP")
    print("R -> Reading data from a csv file")

def read_csv_file():
    FileName = input("Enter csv file name: ")
    return pd.read_csv(FileName)

def print_data_analysis(data):
    print("First  10 rows from csv file: \n")
    print(data.head(n=10))
    print("\n\n Info\n")
    print(data.info())
    print("\n\n describe\n")
    print(data.describe())
    eda.PrintHisto(data)

def DataraProcessing():
    categories.remove('target')
    dataset = pd.get_dummies(Data, columns = categories)
    
    scaler = StandardScaler()
    columns_to_scale = ['age', 'oldpeak', 'chol', 'thalach',  'trestbps']
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

    return dataset



Data = read_csv_file()
''' print_data_analysis(Data) '''
categories = []
continous = []

for i in Data.columns:
    if len(Data[i].unique()) <= 10:
        categories.append(i)
    else:
        continous.append(i)

data_1 = DataraProcessing()

X = data_1.drop(columns=["target"])
y = data_1.target

print(X.head())
print(y.head())








