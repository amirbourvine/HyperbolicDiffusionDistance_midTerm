import numpy as np
from classification import main_divided, patch_to_points, throw_0_labels, main
from preperation import prepare
from utils import hdd, hde
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



def calc_hdd(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    st = time.time()
    distances,P,y_patches,num_patches_in_row, labels_padded = prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
    
    print("PREPARE TIME: ", time.time()-st)
    st = time.time()

    HDE = hde(distances)
    HDE = np.abs(HDE)

    print("HDE TIME: ", time.time()-st)
    st = time.time()

    
    hdd_mat = hdd(HDE, P)

    print("HDD TIME: ", time.time()-st)

    return hdd_mat, labels_padded, num_patches_in_row,y_patches
    
    

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


from numba import cuda

def whole_pipeline_divided_paralel(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    distance_mat_arr = np.ndarray(shape=(X.shape[-1],), dtype=np.ndarray)
    
    threadsperblock = 32 

    # Calculate the number of thread blocks in the grid
    blockspergrid = (distance_mat_arr.shape[0] + (threadsperblock - 1)) // threadsperblock

    distance_mat_arr_dev = cuda.to_device(distance_mat_arr)


    # Now start the kernel
    my_kernel[blockspergrid, threadsperblock](distance_mat_arr_dev, X, y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch)
    
    result_array = distance_mat_arr_dev.copy_to_host()


    X_curr = X[:,:,0].reshape((X.shape[0],X.shape[1],1))
    _, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main_divided(result_array, y_patches, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row)


@cuda.jit
def my_kernel(distance_mat_arr, X, y, rows_factor, cols_factor, is_normalize_each_band, method_label_patch):
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    bx = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    i = tx + bx * bw
    if i < distance_mat_arr.shape[0]:  # Check array boundaries
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i] = d_HDD