from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

def DataProcessing(data,ratio):
    categories = []
    rest = []

    for i in data.columns:
        if len(data[i].unique()) <= 10:
            categories.append(i)
        else:
            rest.append(i)
    categories.remove('target')
    dataset = pd.get_dummies(data, columns = categories)
    
    scaler = StandardScaler()
    columns_to_scale = ['age', 'oldpeak', 'chol', 'thalach',  'trestbps']
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])
    X = dataset.drop(['target'], axis = 1)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    return dataset, X_train, X_test ,y_train, y_test

def PrintScoreTrain(model, X_train, y_train):
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
    return predictions

def PrintScoreTest(model, X_test, y_test):
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
    return predictions

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 20),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def PlotLearningCurve(data, estimator):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    X = data.drop(['target'], axis = 1)
    y = data.target
    
    title = "Learning Curves " + str(estimator)
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )
    plt.show()
