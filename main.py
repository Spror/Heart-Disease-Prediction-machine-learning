import pandas as pd
import Data as DataOperations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import training as TrainingInstructions
from sklearn.tree import DecisionTreeClassifier

def printHelp():
    print("#### HELP ####")
    print("H -> printing HELP")
    print("R -> Reading data from a csv file")


def DataProcessing():
    categories.remove('target')
    dataset = pd.get_dummies(Data, columns = categories)
    
    scaler = StandardScaler()
    columns_to_scale = ['age', 'oldpeak', 'chol', 'thalach',  'trestbps']
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

    return dataset




categories = []
rest = []

# Podzielenie kolumn na kategorie
Data = DataOperations.read_csv_file()
for i in Data.columns:
    if len(Data[i].unique()) <= 10:
        categories.append(i)
    else:
        rest.append(i)

# Modyfikacja danych w celu polepszenia zbioru do nauki
data_1 = DataProcessing()

# Podział danych 
X = data_1.drop(columns=["target"])
y = data_1.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
TrainingInstructions.PrintScore(model, X_train, y_train, X_test, y_test, train=True)
TrainingInstructions.PrintScore(model, X_train, y_train, X_test, y_test, train=False)



