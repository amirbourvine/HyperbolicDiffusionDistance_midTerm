import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_basic_statistics(df):
    print("Has 1 image, 103 spectral bands, 610 * 340 pixels (207400 in total)")
    print("every pixal is labeled between 0 and 9 (clustering ground truth)")
    print("Shape is: ", df.shape)
    print("A few rows of data: \n")
    print(df.sample(5))

def read_dataset(gt=False):
    if gt:
        df = pd.read_csv('paviaU/paviaU_gt.csv')
    else:
        df = pd.read_csv('paviaU/paviaU.csv')

    df.drop(df.columns[0], axis=1, inplace = True)
    
    return df

def plot_gt(df):
    arr = df.to_numpy()
    plt.matshow(arr, cmap=plt.cm.viridis)
    plt.axis('off')
    plt.colorbar()
    plt.show()


def plot_spectral_band(num=5):
    arr_list = []
    for i in range(num):
        c = np.random.randint(103)
        arr = df[f'{c}'].to_numpy()
        arr_list.append((arr.reshape((610,340)),c))

    _, axs = plt.subplots(1,num, sharex=True, sharey=True)

    if num == 1:
        axs  = [axs]

    for i in range(num):
        arr,c = arr_list[i]
        axs[i].matshow(arr, cmap=plt.cm.viridis)
        axs[i].axis('off')
        axs[i].title.set_text(f"Band - {c}")
    plt.show()



df = read_dataset(gt=False)
plot_spectral_band()