import matplotlib.pyplot as plt
import pandas as pd


def read_csv_file(path):
    data = pd.read_csv(path)
    print(data.head())
    return data


def PrintHisto(data):
    data.hist(bins=50, figsize=(20, 15))
    plt.show()


def print_data_analysis(data):
    print("First  10 rows from csv file: \n")
    print(data.head(n=10))
    print("\n\n Info\n")
    print(data.info())
    print("\n\n describe\n")
    print(data.describe())


def EnterRowToPredict():

    print("Enter data to make prediction:")
    age = abs(int(input('Enter age: ')))

    sex = int(input('Enter sex (1 = male; 0 = female): '))
    if sex != 0 and sex != 1:
        print("sex must be in 0-1 range")
        quit()

    cp = int(input('Enter cp (0-3): '))
    if cp > 3 or cp < 0:
        print("cp must be in 0-3 range")
        quit()

    trestbps = abs(int(input("Enter resting blood pressure: ")))
    chol = abs(int(input("Enter serum cholestoral in mg/dl: ")))
    fbs = abs(int(input("Enter fasting blood sugar: ")))

    restecg = int(input('Enter resting electrocardiographic results (0-2): '))
    if restecg > 2 or restecg < 0:
        print("restecg must be in 0-2 range")
        quit()

    thalach = abs(int(input("Enter maximum heart rate achieved: ")))

    exang = int(input("Enter exercise induced angina (1 = yes; 0 = no): "))
    if exang != 0 and exang != 1:
        print("exang must be in 0-1 range")
        quit()

    oldpeak = abs(int(input("Enter ST depression : ")))

    slope = int(input(
        "Enter the slope of the peak exercise ST segment (0: Upsloping, 1: Flatsloping, 2: Downslopins ): "))
    if slope > 2 or slope < 0:
        print("slope must be in 0-2 range")
        quit()

    ca = int(input("Enter number of major vessels (0-4) colored by flourosopy: "))
    if ca > 4 or ca < 0:
        print("ca must be in 0-4 range")
        quit()

    thal = int(input("Enter thalium stress result (0-3): "))
    if thal > 3 or thal < 0:
        print("thal must be in 0-3 range")
        quit()

    print("Your entered data:\nage: " + str(age))
    print("sex: " + str(sex))
    print("cp: " + str(cp))
    print("trestbps: " + str(trestbps))
    print("chol: " + str(chol))
    print("fbs: " + str(fbs))
    print("restecg: " + str(restecg))
    print("thalach: " + str(thalach))
    print("exang: " + str(exang))
    print("oldpeak: " + str(oldpeak))
    print("slope: " + str(slope))
    print("ca: " + str(ca))
    print("thal: " + str(thal))
    
    sex_ = [0, 0]
    if sex == 0:
        sex_[0] = 1
    else:
        sex_[1] = 1

        
    cp_ = [0, 0, 0, 0]
    if cp == 0:
        cp_[0] = 1
    elif cp == 1:
        cp_[1] = 1
    elif cp == 2:
        cp_[2] = 1
    else:
        cp_[3] = 1

    fbs_ = [0, 0]
    if fbs == 0:
        fbs_[0] = 1
    else:
        fbs_[1] = 1


    restecg_ = [0, 0, 0]
    if restecg == 0:
        restecg_[0] = 1
    elif restecg == 1:
        restecg_[1] = 1
    else:
        restecg_[2] = 1

    exang_ = [0, 0]
    if exang == 0:
        exang_[0] = 1
    else:
        exang_[1] = 1


    slope_ = [0, 0, 0]
    if slope == 0:
        slope_[0] = 1
    elif slope == 1:
        slope_[1] = 1
    else:
        slope_[2] = 1

    ca_ = [0, 0, 0, 0, 0]
    if ca == 0:
        ca_[0] = 1
    elif ca == 1:
        ca_[1] = 1
    elif ca ==2:
        ca_[2] = 1
    elif ca ==3:
        ca_[3] = 1
    else:
        ca_[4] = 1

    thal_ = [0, 0, 0, 0]
    if thal == 0:
        thal_[0] = 1
    elif thal == 1:
        thal_[1] = 1
    elif thal ==2:
        thal_[2] = 1
    else:
        thal_[3] = 1


    d = {'age':[age], 'sex_0':[sex_[0]], 'sex_1':[sex_[1]], 'cp_0':[cp_[0]], 'cp_1':[cp_[1]], 'cp_2':[cp_[2]],
         'cp_3':[cp_[3]], 'trestbps':[trestbps], 'chol':[chol], 'fbs_0':[fbs_[0]], 'fbs_1':[fbs_[1]],
         'restecg_0':[restecg_[0]], 'restecg_1':[restecg_[1]], 'restecg_2':[restecg_[2]], 'thalach':[thalach],
         'exang_0':[exang_[0]], 'oldpeak':[oldpeak], 'slope_0': [slope_[0]], 'slope_1': [slope_[1]], 'slope_2': [slope_[2]],
         'ca_0':[ca_[0]], 'ca_1':[ca_[1]], 'ca_2':[ca_[2]], 'ca_3':[ca_[3]], 'ca_4':[ca_[4]], 'thal_0':[thal_[0]],
         'thal_1':[thal_[1]], 'thal_2':[thal_[2]], 'thal_3':[thal_[3]]}

    row_to_predict = pd.DataFrame(data=d)


    return row_to_predict
