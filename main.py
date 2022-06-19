import argparse
import csv
from sklearn.preprocessing import StandardScaler
import Data as data
import pandas as pd
import training as tra
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


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
    Scaler = None

    if path:
        try:
            print(path)
            csv_file = data.read_csv_file(path)
        except:
            print("FILE WITH THAT NAME DOES NOT EXIST")
            print("----------------------------------\n")
    
    if histograms:
        #try:
        data.PrintHisto(csv_file)
        #except:
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
            ration = float(input('Enter test size (0-1): '))
            processed_data, X_train, X_test, y_train, y_test, Scaler = tra.DataProcessing(csv_file, ration)
            
            if learn_model == 'tree': 
                params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
                tree = DecisionTreeClassifier(random_state=42)
                model = GridSearchCV(tree, params)
                
            elif learn_model == 'SVC':
                params = {'kernel':('rbf', 'poly', 'sigmoid'), 'C':[1,5,10,15,20,25,30,35]}
                svc = SVC()
                model = GridSearchCV(svc, params)
                

            elif learn_model =='kne':
                params = {'n_neighbors': [3, 5, 11, 19], 'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']}
                Kn = KNeighborsClassifier()
                model = GridSearchCV(Kn, params)
                
            model.fit(X_train, y_train)
            print("Best params:" + str(model.best_params_))
            model = model.best_estimator_
            tra.PrintScoreTest(model, X_train, y_train)
            tra.PrintScoreTrain(model, X_test, y_test)
            tra.PlotLearningCurve(processed_data, model)

        except:
            print("YOU NEED ADD A PATH TO FILE --> -p <path>")
            print("----------------------------------\n")

    if save_model:
        if learn_model: 
            try:
                pipe = Pipeline([('scaler', Scaler), ('estimator', model)])
                dump(pipe, 'Models/' + save_model + ".joblib")
                print("saved as 'Models/" + save_model + ".joblib")
                

            except:
                print("YOU NEED ADD A PATH TO FILE --> -p <path>")
                print("----------------------------------\n")

    if load_model and path:
        try:
            pipe = load('Models/' + load_model + '.joblib')
            model = pipe[1]
            print("\n Loaded model:")
            print(model)
            categories = []
            rest = []

            for i in csv_file.columns:
                if len(csv_file[i].unique()) <= 10:
                    categories.append(i)
                else:
                    rest.append(i)
        

        
            columns_to_scale = ['age', 'oldpeak', 'chol', 'thalach',  'trestbps']
            csv_file[columns_to_scale] = pipe[0].fit_transform(csv_file[columns_to_scale])
            categories.remove('target')
            dataset = pd.get_dummies(csv_file, columns = categories)
            

            X = dataset.drop(['target'], axis = 1)
            y = dataset.target
            

            temp_csv = csv_file
            temp_csv.drop(['target'], axis = 1)
            predictions = tra.PrintScoreTest(model, X, y)
            
            X_with_predictions = temp_csv.assign(Target = predictions)
            X_with_predictions.to_csv(path_or_buf= "Predictions/predictions.csv")

        except:
            print("PROBABLY WRONG FILE NAME")
            print("----------------------------------\n")


        
    elif load_model:
        try:
            row_to_predict =  data.EnterRowToPredict()

            pipe = load('Models/' + load_model + '.joblib')
            model = pipe[1]
            print("\n Loaded model:")
            print(model)
            

            columns_to_scale = ['age', 'oldpeak', 'chol', 'thalach',  'trestbps']
            row_to_predict[columns_to_scale] = pipe[0].fit_transform(row_to_predict[columns_to_scale])
            prediction = model.predict(row_to_predict)
            print("Wynik predykcji (Target): " + str(prediction))
            
        except:
            print("PROBABLY WRONG FILE NAME")
            print("----------------------------------\n")


