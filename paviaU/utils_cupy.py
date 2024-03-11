import cupy as cp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from paviaU.preperation import prepare
from utils import CONST_K,ALPHA,TOL,CONST_C, hdd_try, hde
from classification import main_divided, main







def hdd_cupy(X,P):
    d_HDD = cp.zeros_like(P)

    for k in range(CONST_K + 1):
        norms = cp.scipy.spatial.distance_matrix(X[k], X[k])
        sum_matrix = 2 * cp.arcsinh((2 ** (-k * ALPHA + 1)) * norms)
        d_HDD += sum_matrix

    return d_HDD








def svd_symmetric_cupy(M):
  s,u = cp.linalg.eigh(M)  #eigenvalues and eigenvectors

  u = u[:, cp.argsort(s)[::-1]]
  s = (cp.sort(s)[::-1])

  v = u.copy()
  v[:,s<0] = -u[:,s<0] #replacing the corresponding columns with negative sign

  s = cp.absolute(s)

  s = cp.where(s>TOL, s, TOL)

  return u, s, v.T


def calc_svd_p_cupy(d):
  epsilon = CONST_C*cp.median(d)
  W = cp.exp(-1*d/epsilon)
  S_vec = cp.sum(W,axis=1)

  S = cp.diag(1/S_vec)
  W_gal = cp.matmul(cp.matmul(S,W),S)

  D_vec = cp.sum(W_gal,axis=1)

  D_minus_half = cp.diag(1 / cp.sqrt(D_vec))
  D_plus_half = cp.diag(cp.sqrt(D_vec))


  M = cp.matmul(cp.matmul(D_minus_half,W_gal),D_minus_half)

  U,S,UT = svd_symmetric_cupy(M)

  return (cp.matmul(D_minus_half,U)),(S),(cp.matmul(UT,D_plus_half))



def hde_cupy(shortest_paths_mat):
  U, S, Vt = calc_svd_p_cupy(shortest_paths_mat)

  X = cp.zeros((CONST_K + 1, shortest_paths_mat.shape[0], shortest_paths_mat.shape[1] + 1), dtype=cp.complex64)
  S_keep=S
  for k in range (0, CONST_K + 1):
    S = cp.float_power(S_keep, 2 ** (-k))

    aux = cp.matmul(cp.matmul(U,cp.diag(S)),Vt)

    aux = cp.transpose(cp.sqrt((cp.where(aux > TOL, aux, TOL))))
    X[k] = cp.transpose(cp.concatenate((aux, cp.full(shortest_paths_mat.shape[0], 2 ** (k * ALPHA - 2)).reshape(1, -1)), axis=0))

  return X







def padWithZeros_cupy(X, left_margin, right_margin, top_margin, bottom_margin, dim=3):
    if dim == 3:
        newX = cp.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin, X.shape[2]))
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin, :] = X
    
    elif dim == 2:
        newX = cp.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin))
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin] = X

    else:
        newX = []

    return newX

def calc_patch_label_cupy(labels, i, j, rows_factor, cols_factor, method='center'):
    if method=='center':
        return labels[i*rows_factor + rows_factor//2, j*cols_factor + cols_factor//2]
    elif method=='most_common':
        labels_patch = (labels[i*rows_factor : (i+1)*rows_factor, j*cols_factor : (j+1)*cols_factor]).astype(int)
        counts = cp.bincount(labels_patch.flatten())

        # in order to not let 0 values take over and set many labels to 0 which leads to small number of non zero labeled patches
        counts[0]=1

        return cp.argmax(counts)
    
    print("ERROR- INCORRECT METHOD FOR LABELING PATCHES")

def patch_data_cupy(data, labels, rows_factor, cols_factor, method_label_patch):
    rows, cols, channels = data.shape

    left_margin = ((-rows) % rows_factor) // 2
    right_margin = ((-rows) % rows_factor + 1) // 2
    top_margin = ((-cols) % cols_factor) // 2
    bottom_margin = ((-cols) % cols_factor + 1) // 2

    data = padWithZeros_cupy(data, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin)
    labels = padWithZeros_cupy(labels, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin, dim=2)

    new_rows, new_cols, _ = data.shape

    patched_data = cp.empty((new_rows // rows_factor, new_cols // cols_factor, rows_factor, cols_factor, channels))
    patched_labels = cp.zeros((patched_data.shape[0], patched_data.shape[1]))

    for i in range(new_rows // rows_factor):
        for j in range(new_cols // cols_factor):
            datapoint = data[i*rows_factor: (i+1)*rows_factor, j*cols_factor: (j+1)*cols_factor, :]
            patched_data[i, j] = datapoint
            patched_labels[i, j] = calc_patch_label_cupy(labels, i, j, rows_factor, cols_factor, method=method_label_patch)


    return patched_data, patched_labels, labels


def normalize_each_band_cupy(X):
    X_normalized = cp.zeros_like(X)
    for i in range(X.shape[2]):
        X_band = X[:,:,i]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(X_band)
        X_normalized[:,:,i] = scaled_data

    return X_normalized

import time

def calc_P_cupy(d, apply_2_norm=False):
  epsilon = CONST_C*cp.median(d)
  W = cp.exp(-1*d/epsilon)

  if apply_2_norm:
    S_vec = cp.sum(W,axis=1)

    S = cp.diag(1/S_vec)
    
    W_gal = cp.matmul(cp.matmul(S,W),S)
    

    D_vec = cp.sum(W_gal,axis=1)
    D = cp.diag(1 / D_vec)
    P = cp.matmul(D,W_gal)

  else:
    D_vec = cp.sum(W,axis=1)
    D = cp.diag(1/ D_vec)
    P = cp.matmul(D,W)

  return P


def prepare_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    # print("$$$$$$$$$$ IN PREPERATION $$$$$$$$$$$$")

    # st = time.time()

    if is_normalize_each_band:
        X = normalize_each_band_cupy(X)

    # print("NORMALIZATION: ", time.time()-st)
    # st = time.time()

    X_patches, y_patches, labels_padded= patch_data_cupy(X, y, rows_factor, cols_factor, method_label_patch)
    
    # print("PATCHING: ", time.time()-st)
    # st = time.time()

    num_patches_in_row = y_patches.shape[1]

    y_patches = y_patches.flatten()

    X_patches = X_patches.reshape(-1, cp.prod(X_patches.shape[2:]))

    # st = time.time()

    distances = cp.scipy.spatial.distance_matrix(X_patches, X_patches)
    
    # print("DISTANCES WITH CDIST: ", time.time()-st)
    # st = time.time()
    

    P = calc_P_cupy(distances, apply_2_norm=True)

    # print("CALC_P: ", time.time()-st)

    return distances,P,y_patches,num_patches_in_row, labels_padded









def calc_hdd_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    st = time.time()
    distances,P,y_patches,num_patches_in_row, labels_padded = prepare_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
    
    print("PREPARE TIME: ", time.time()-st)
    st = time.time()

    HDE = hde_cupy(distances)
    HDE = cp.abs(HDE)

    print("HDE TIME: ", time.time()-st)
    st = time.time()

    # print("HDE.shape: ", HDE.shape)
    
    hdd_mat = hdd_cupy(HDE, P)

    # hdd_mat_2 = hdd(HDE, P)
    # print("NORM: ", np.linalg.norm(hdd_mat-hdd_mat_2))

    print("HDD TIME: ", time.time()-st)

    return hdd_mat, labels_padded, num_patches_in_row,y_patches



def whole_pipeline_all_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    print("XXXXXXX IN METHOD XXXXXXXXX")
    st = time.time()
    d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    print("WHOLE METHOD TIME: ", time.time()-st)
    st = time.time()

    print("XXXXXXX IN CLASSIFICATION XXXXXXXXX")
    n_neighbors = 3

    y_patches = y_patches.astype(int)
    
    main(cp.asnumpy(d_HDD), cp.asnumpy(y_patches), n_neighbors, cp.asnumpy(labels_padded), rows_factor, cols_factor, num_patches_in_row)

    print("WHOLE CLASSIFICATION TIME: ", time.time()-st)




def whole_pipeline_divided_cupy(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center', is_print=False):
    st = time.time()
    
    distance_mat_arr = cp.ndarray(shape=(X.shape[-1],), dtype=cp.ndarray)
    for i in range(X.shape[-1]):
        if is_print:
            print((i+1)," out of: ", X.shape[-1])
        X_curr = X[:,:,i].reshape((X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd_cupy(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i] = d_HDD
    

    print("TOTAL TIME FOR METHOD: ", time.time()-st)

    n_neighbors = 3

    y_patches = y_patches.astype(int)

    main_divided(cp.asnumpy(distance_mat_arr), cp.asnumpy(y_patches), n_neighbors, cp.asnumpy(labels_padded), rows_factor, cols_factor, num_patches_in_row)



from initial_plots import read_dataset

def test_cupy_implementation():
    df = read_dataset(gt=False)

    X = np.array(df)
    X = X.reshape((610,340, 103))

    df = read_dataset(gt=True)
    y = np.array(df)

    rows_factor=21
    cols_factor=21

    distances,P,y_patches,num_patches_in_row, labels_padded = prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center')
    distances_c,P_c,y_patches_c,num_patches_in_row_c, labels_padded_c = prepare_cupy(cp.asarray(X),cp.asarray(y), rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center')

    print("NORM DISTANCES: ", np.linalg.norm(distances-cp.asnumpy(distances_c)))
    print("NORM P: ", np.linalg.norm(P-cp.asnumpy(P_c)))
    print("NORM y_patches: ", np.linalg.norm(y_patches-cp.asnumpy(y_patches_c)))
    print("NORM labels_padded: ", np.linalg.norm(labels_padded-cp.asnumpy(labels_padded_c)))
    print("DIFF num_patches_in_row: ", (num_patches_in_row-num_patches_in_row_c))

    HDE = hde(distances)
    HDE = np.abs(HDE)

    HDE_c = hde_cupy(distances_c)
    HDE_c = cp.abs(HDE_c)

    print("NORM HDE: ", np.linalg.norm(HDE-cp.asnumpy(HDE_c)))
    
    hdd_mat = hdd_try(HDE, P)
    hdd_mat_C = hdd_cupy(HDE_c, P_c)

    print("NORM HDD: ", np.linalg.norm(hdd_mat-cp.asnumpy(hdd_mat_C)))
