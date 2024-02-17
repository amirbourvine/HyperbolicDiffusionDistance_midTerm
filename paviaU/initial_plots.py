import pandas as pd


def print_basic_statistics(df):
    print("Shape is: ", df.shape)
    print("Has 103 spectral bands")
    print("Has 610 * 340 pixels (207400 in total)")
    

print(df.head(5))

def read_dataset():
    df = pd.read_csv('paviaU.csv')

    df.drop(df.columns[0], axis=1, inplace = True)

    return df
