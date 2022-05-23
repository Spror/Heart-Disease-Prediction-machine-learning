from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def PrintScore(model, X_train, y_train, X_test, y_test, train=True):
    if train:
        predictions = model.predict(X_train)
        report = pd.DataFrame(classification_report(y_train, predictions, output_dict=True))
        print("Results of training:\n")
        print("######################\n")
        print("Accuracy: " + str(accuracy_score(y_train, predictions) * 100) + "%\n")
        print("----------------------------\n")
        print("Report:\n" + str(report))
        print("----------------------------\n")
        print("Confusion Matrix:\n" + str(confusion_matrix(y_train, predictions)))
        print("######################\n")

    elif train == False:
        predictions = model.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True))
        print("Test result:\n")
        print("######################\n")
        print("Accuracy: " + str(accuracy_score(y_test, predictions) * 100) + "%\n")
        print("----------------------------\n")
        print("Report:\n" + str(report))
        print("----------------------------\n")
        print("Confusion Matrix:\n" + str(confusion_matrix(y_test, predictions)))
        print("######################\n")






