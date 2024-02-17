import pandas as pd
import matplotlib.pyplot as plt

def print_basic_statistics(df):
    print("Has 1 image, 103 spectral bands, 610 * 340 pixels (207400 in total)")
    print("every pixal is labeled between 0 and 9 (clustering ground truth)")
    print("Shape is: ", df.shape)
    print("A few rows of data: \n")
    print(df.sample(5))

def read_dataset():
    df = pd.read_csv('paviaU/paviaU.csv')

    df.drop(df.columns[0], axis=1, inplace = True)

    return df

def plot_spectral_band(index, num=1):
    arr_list = []
    for i in range(index, index+num):
        arr = df[f'{i}'].to_numpy()
        arr_list.append(arr.reshape((610,340)))

    _, axs = plt.subplots(1,num, sharex=True, sharey=True)

    if num == 1:
        axs  = [axs]

    for i in range(num):
        axs[i].matshow(arr_list[i], cmap=plt.cm.viridis)
    plt.show()


df = read_dataset()
# print_basic_statistics(df)

plot_spectral_band(95)
# plot_spectral_band(102)