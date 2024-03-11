import numpy as np
from scipy.spatial.distance import cdist

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import *


from sklearn.preprocessing import MinMaxScaler
import matplotlib

# matplotlib.use('TkAgg')

def padWithZeros(X, left_margin, right_margin, top_margin, bottom_margin, dim=3):
    if dim == 3:
        newX = np.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin, X.shape[2]))
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin, :] = X
    
    elif dim == 2:
        newX = np.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin))
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin] = X

    else:
        newX = []

    return newX

def calc_patch_label(labels, i, j, rows_factor, cols_factor, method='center'):
    if method=='center':
        return labels[i*rows_factor + rows_factor//2, j*cols_factor + cols_factor//2]
    elif method=='most_common':
        labels_patch = (labels[i*rows_factor : (i+1)*rows_factor, j*cols_factor : (j+1)*cols_factor]).astype(int)
        counts = np.bincount(labels_patch.flatten())

        # in order to not let 0 values take over and set many labels to 0 which leads to small number of non zero labeled patches
        counts[0]=1

        return np.argmax(counts)
    
    print("ERROR- INCORRECT METHOD FOR LABELING PATCHES")

def patch_data(data, labels, rows_factor, cols_factor, method_label_patch):
    rows, cols, channels = data.shape

    left_margin = ((-rows) % rows_factor) // 2
    right_margin = ((-rows) % rows_factor + 1) // 2
    top_margin = ((-cols) % cols_factor) // 2
    bottom_margin = ((-cols) % cols_factor + 1) // 2

    data = padWithZeros(data, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin)
    labels = padWithZeros(labels, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin, dim=2)

    new_rows, new_cols, _ = data.shape

    patched_data = np.empty((new_rows // rows_factor, new_cols // cols_factor, rows_factor, cols_factor, channels))
    patched_labels = np.zeros((patched_data.shape[0], patched_data.shape[1]))

    for i in range(new_rows // rows_factor):
        for j in range(new_cols // cols_factor):
            datapoint = data[i*rows_factor: (i+1)*rows_factor, j*cols_factor: (j+1)*cols_factor, :]
            patched_data[i, j] = datapoint
            patched_labels[i, j] = calc_patch_label(labels, i, j, rows_factor, cols_factor, method=method_label_patch)


    return patched_data, patched_labels, labels


def normalize_each_band(X):
    X_normalized = np.zeros_like(X,dtype=float)
    for i in range(X.shape[2]):
        X_band = X[:,:,i]
        scaled_data = (X_band - np.min(X_band)) / (np.max(X_band) - np.min(X_band))
        X_normalized[:,:,i] = scaled_data

    return X_normalized

import time

def prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    # print("$$$$$$$$$$ IN PREPERATION $$$$$$$$$$$$")

    # st = time.time()

    if is_normalize_each_band:
        X = normalize_each_band(X)

    # print("NORMALIZATION: ", time.time()-st)
    # st = time.time()

    X_patches, y_patches, labels_padded= patch_data(X, y, rows_factor, cols_factor, method_label_patch)
    
    # print("PATCHING: ", time.time()-st)
    # st = time.time()

    num_patches_in_row = y_patches.shape[1]

    y_patches = y_patches.flatten()

    X_patches = X_patches.reshape(-1, np.prod(X_patches.shape[2:]))

    # st = time.time()

    distances = cdist(X_patches, X_patches, 'euclidean')
    
    # print("DISTANCES WITH CDIST: ", time.time()-st)
    # st = time.time()
    

    P = calc_P(distances, apply_2_norm=True)

    # print("CALC_P: ", time.time()-st)

    return distances,P,y_patches,num_patches_in_row, labels_padded

def figure_B(distances, rows_factor, cols_factor):
    import math

    colors,norm,cmap = multi_scale_propagated_densities_colors(distances, i=(260), k=0)

    x_points = np.zeros((distances.shape[0]))
    y_points = np.zeros((distances.shape[1]))

    rows = math.ceil(610 / rows_factor)
    cols = math.ceil(340 / cols_factor)
    for i in range(rows):
        for j in range(cols):
            x_points[i * cols + j] = j
            y_points[i * cols + j] = i

    plt.scatter(x_points, y_points, c=colors, norm=norm, cmap=cmap,edgecolors='black',s=60)

    plt.colorbar()

    plt.gca().set_aspect('equal')

    plt.show()