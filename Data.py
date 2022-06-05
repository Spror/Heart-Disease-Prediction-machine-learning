import matplotlib.pyplot as plt
import pandas as pd

def read_csv_file(path):
    data = pd.read_csv(path)
    print(data.head())
    return data

def PrintHisto(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()

def print_data_analysis(data):
    print("First  10 rows from csv file: \n")
    print(data.head(n=10))
    print("\n\n Info\n")
    print(data.info())
    print("\n\n describe\n")
    print(data.describe())
    


