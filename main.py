import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hvplot.pandas
from scipy import stats


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


data = pd.read_csv("/home/wiktor/Desktop/MGR/ML/heart.csv")

print(data.head())
print(data.info())
print(data.shape)