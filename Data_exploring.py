import matplotlib.pyplot as plt

def PrintHisto(data):
    data.hist(bins=50, figsize=(20,15))
    plt.show()
   