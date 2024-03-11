import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import earthpy.plot as ep

import os

def print_basic_statistics(df):
    print("Has 1 image, 103 spectral bands, 610 * 340 pixels (207400 in total)")
    print("every pixal is labeled between 0 and 9 (clustering ground truth)")
    print("Shape is: ", df.shape)
    print("A few rows of data: \n")
    print(df.sample(5))

def read_dataset(gt=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if gt:
        df = pd.read_csv(os.path.join(current_dir,'paviaU_gt.csv'))
    else:
        df = pd.read_csv(os.path.join(current_dir,'paviaU.csv'))

    df.drop(df.columns[0], axis=1, inplace = True)
    
    return df

def plot_gt(df):
    arr = df.to_numpy()
    plt.matshow(arr, cmap=plt.cm.viridis)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def plot_composite(df):
    data = df.to_numpy().reshape(((610, 340,103)))
    data = np.moveaxis(data, 2, 0)
    ep.plot_rgb(data, rgb=(36, 17, 11), title='Composite Image of Pavia University', figsize=(10, 8))
    plt.show()

def plot_spectral_band(df, num=4):
    arr_list = []
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens',  'Reds']
    for i in range(num):
        c = np.random.randint(103)
        arr = df[f'{c}'].to_numpy()
        arr_list.append((arr.reshape((610,340)),c))

    _, axs = plt.subplots(1,num, sharex=True, sharey=True, figsize=(15, 15))
    plt.tight_layout(pad=5.0)
    
    if num == 1:
        axs  = [axs]

    for i in range(num):
        arr,c = arr_list[i]
        im = axs[i].matshow(arr, cmap=cmaps[i%len(cmaps)])
        axs[i].axis('off')
        axs[i].title.set_text(f"Band - {c}")
        plt.colorbar(im,ax=axs[i],fraction=0.08, pad=0.04)
    plt.show()

