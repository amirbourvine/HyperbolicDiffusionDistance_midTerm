import numpy as np
from classification import main_divided, patch_to_points, throw_0_labels, main
from preperation import prepare
from utils import hdd, hde

def whole_pipeline_all(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main(d_HDD, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)



def calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    distances,P,y_patches,num_patches_in_row, labels_padded = prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
    HDE = hde(distances)
    HDE = np.abs(HDE)
    return hdd(HDE, P), labels_padded, num_patches_in_row,y_patches
    
    

def whole_pipeline_divided(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    distance_mat_arr = np.ndarray(shape=(X.shape[-1],), dtype=np.ndarray)
    for i in range(X.shape[-1]):
        print((i+1)," out of: ", X.shape[-1])
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i] = d_HDD
    

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main_divided(distance_mat_arr, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)


def check(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    stam = np.ndarray((1,),dtype=np.ndarray)
    stam[0] = d_HDD

    main_divided(stam, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)



def whole_pipeline_divided_paralel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    distance_mat_arr = np.ndarray(shape=(X.shape[-1],), dtype=np.ndarray)
    for i in range(X.shape[-1]):
        print((i+1)," out of: ", X.shape[-1])
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i] = d_HDD
    

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main_divided(distance_mat_arr, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)