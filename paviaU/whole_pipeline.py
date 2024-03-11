import multiprocessing
import numpy as np
from classification import main_divided, patch_to_points, throw_0_labels, main
from preperation import prepare
from utils import hdd, hdd_try, hde
import time

def whole_pipeline_all(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    print("XXXXXXX IN METHOD XXXXXXXXX")
    st = time.time()
    d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    print("WHOLE METHOD TIME: ", time.time()-st)
    st = time.time()

    print("XXXXXXX IN CLASSIFICATION XXXXXXXXX")
    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main(d_HDD, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)

    print("WHOLE CLASSIFICATION TIME: ", time.time()-st)

def calc_hdd_for_multiprocessing(X,y, rows_factor, cols_factor):
    distances,P,_,_, _ = prepare(X,y, rows_factor, cols_factor)

    HDE = hde(distances)
    HDE = np.abs(HDE)

    hdd_mat = hdd_try(HDE, P)

    return hdd_mat

def calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    st = time.time()
    distances,P,y_patches,num_patches_in_row, labels_padded = prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
    
    print("PREPARE TIME: ", time.time()-st)
    st = time.time()

    HDE = hde(distances)
    HDE = np.abs(HDE)

    print("HDE TIME: ", time.time()-st)
    st = time.time()

    # print("HDE.shape: ", HDE.shape)
    
    hdd_mat = hdd_try(HDE, P)

    # hdd_mat_2 = hdd(HDE, P)
    # print("NORM: ", np.linalg.norm(hdd_mat-hdd_mat_2))

    print("HDD TIME: ", time.time()-st)

    return hdd_mat, labels_padded, num_patches_in_row,y_patches
    
    

def whole_pipeline_divided(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center', is_print=False):
    st = time.time()
    
    distance_mat_arr = np.ndarray(shape=(X.shape[-1],), dtype=np.ndarray)
    for i in range(X.shape[-1]):
        if is_print:
            print((i+1)," out of: ", X.shape[-1])
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i] = d_HDD
    

    print("TOTAL TIME FOR METHOD: ", time.time()-st)

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main_divided(distance_mat_arr, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)



def whole_pipeline_divided_parallel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    st = time.time()
    
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    distance_mat_arr = np.ndarray(shape=(X.shape[-1],), dtype=np.ndarray)
    for i in range(X.shape[-1]):
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        tup = (X_curr,y, rows_factor, cols_factor)
        for result in pool.starmap(calc_hdd_for_multiprocessing, (tup,)):
            distance_mat_arr[i] = result

    pool.close()  # no more tasks

    X_curr = X[:,:,0].reshape((X.shape[0],X.shape[1],1))
    _, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    pool.join()  # wrap up current tasks

    print("TOTAL TIME FOR METHOD: ", time.time()-st)

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
