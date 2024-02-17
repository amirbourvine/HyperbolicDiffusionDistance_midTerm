import pandas as pd


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

def plot_spectral_band(index):
    arr = df[f'{index}'].to_numpy()
    arr = arr.reshape((610,340))
    print(arr.shape)


df = read_dataset()
# print_basic_statistics(df)

plot_spectral_band(0)