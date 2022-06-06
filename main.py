import argparse
from sklearn.preprocessing import StandardScaler
import Data as data
import pandas as pd
import training as tra
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

if __name__ == '__main__':
    ap=parser = argparse.ArgumentParser(description='Heart prediction machine learning with scikit')
    ap.add_argument("-p", "--path", required=False, help="csv file path", type=str)
    ap.add_argument("-his", "--histograms", action='store_true', help="Shows various histograms of data from a csv file ")
    ap.add_argument("-m", "--more_text_info", action='store_true', help="Shows more text info about data from a csv file")
    ap.add_argument("-l", "--learn_model", choices=['tree', 'SVC', 'kne'], help="Learns a model by the chosen algorithm")
    ap.add_argument("-s", "--save_model", required=False, help="Saves learning model. Requires file name which will store model", type=str)
    ap.add_argument("-lm", "--load_model", required=False, help="Loads model from a joblib file", type=str)
    args = vars(ap.parse_args())
    


    path = args["path"]
    histograms = args["histograms"]
    more_text_info = args["more_text_info"]
    learn_model = args['learn_model']
    save_model = args['save_model']
    load_model = args['load_model']

    csv_file = None
    model = None
    predictions = None

    if path:
        try:
            print(path)
            csv_file = data.read_csv_file(path)
        except:
            print("FILE WITH THAT NAME DOES NOT EXIST")
            print("----------------------------------\n")
    
    if histograms:
        try:
            data.PrintHisto(csv_file)
        except:
            print("YOU NEED ADD A PATH TO FILE --> -p <path>")
            print("----------------------------------\n")

    if more_text_info:
        try:
            data.print_data_analysis(csv_file)
        except:
            print("YOU NEED ADD A PATH TO FILE --> -p <path>")
            print("----------------------------------\n")

    if learn_model:
        try:
            ration = 0.2
            processed_data, X_train, X_test, y_train, y_test = tra.DataProcessing(csv_file, ration)
            
            if learn_model == 'tree': 
                model = DecisionTreeClassifier(random_state=42)
                
            elif learn_model == 'SVC':
                model = SVC(kernel='rbf', gamma=0.1, C=1.0)

            elif learn_model =='kne':
                model = KNeighborsClassifier()
                
            model.fit(X_train, y_train)

            tra.PrintScoreTest(model, X_train, y_train)
            tra.PrintScoreTrain(model, X_test, y_test)
            tra.PlotLearningCurve(processed_data, model)

        except:
            print("YOU NEED ADD A PATH TO FILE --> -p <path>")
            print("----------------------------------\n")

    if save_model:
        if learn_model: 
            try:
                dump(model, 'Models/' + save_model + ".joblib")

            except:
                print("YOU NEED ADD A PATH TO FILE --> -p <path>")
                print("----------------------------------\n")

    if load_model and path:
        try:
            model = load('Models/' + load_model + '.joblib')
            print(model)
            categories = []
            rest = []

            for i in csv_file.columns:
                if len(csv_file[i].unique()) <= 10:
                    categories.append(i)
                else:
                    rest.append(i)
            categories.remove('target')
            dataset = pd.get_dummies(csv_file, columns = categories)
            

            X = dataset.drop(['target'], axis = 1)
            y = dataset.target
            #temp_csv = csv_file
            #temp_csv.drop(['target'], axis = 1)
            predictions = tra.PrintScoreTest(model, X, y)
            
            X_with_predictions = X.assign(Target = predictions)
            X_with_predictions.to_csv(path_or_buf= "Predictions/predictions.csv")
            print("Saved as Predictions/predictions.csv")

        except:
            print("PROBABLY WRONG FILE NAME")
            print("----------------------------------\n")
            
    elif load_model:
           # try:
        model = load('Models/' + load_model + '.joblib')
        print(model)
        row_to_predict = data.EnterRowToPredict()
        print(row_to_predict.head())
        predictions = model.predict(row_to_predict)
        print(predictions)

            
        #except:
        print("PROBABLY WRONG FILE NAME")
        print("----------------------------------\n")


        


