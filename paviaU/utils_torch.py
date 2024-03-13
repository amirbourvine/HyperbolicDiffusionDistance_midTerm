import numpy as np
import torch
from preperation import prepare

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import CONST_K,ALPHA,TOL,CONST_C, hdd_try, hde
from classification import main_divided, main



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ", device)


def hdd_torch(X,P):
    d_HDD = torch.zeros_like(P, device=device)

    for k in range(CONST_K + 1):
        norms = torch.cdist(X[k], X[k])
        sum_matrix = 2 * torch.arcsinh((2 ** (-k * ALPHA + 1)) * norms)
        d_HDD += sum_matrix

        del norms
        del sum_matrix

    return d_HDD








def svd_symmetric_torch(M):
  s,u = torch.linalg.eigh(M)  #eigenvalues and eigenvectors

  s, indices = torch.sort(s, descending=True)
  u = u[:, indices]

  v = u.clone()
  v[:,s<0] = -u[:,s<0] #replacing the corresponding columns with negative sign

  torch.abs(s, out=s)

  torch.where(s>TOL, s, torch.tensor([TOL], device=device), out=s)

  return u, s, torch.t(v)


def calc_svd_p_torch(d):
  epsilon = CONST_C*torch.median(d)
  W = torch.exp(-1*d/epsilon)
  S_vec = torch.sum(W,dim=1)

  S = torch.diag(1/S_vec)
  W_gal = torch.matmul(torch.matmul(S,W),S)

  D_vec = torch.sum(W_gal,dim=1)

  D_minus_half = torch.diag(1 / torch.sqrt(D_vec))
  D_plus_half = torch.diag(torch.sqrt(D_vec))


  M = torch.matmul(torch.matmul(D_minus_half,W_gal),D_minus_half)

  del S

  U,S,UT = svd_symmetric_torch(M)

  res = (torch.matmul(D_minus_half,U)),(S),(torch.matmul(UT,D_plus_half))

  del W
  del S_vec
  del S
  del W_gal
  del D_vec
  del D_minus_half
  del D_plus_half
  del M
  del U
  del UT

  return res



def hde_torch(shortest_paths_mat):
  U, S_keep, Vt = calc_svd_p_torch(shortest_paths_mat)
  
  U = U.double()
  S_keep = S_keep.double()
  Vt = Vt.double()

  X = torch.zeros((CONST_K + 1, shortest_paths_mat.shape[0], shortest_paths_mat.shape[1] + 1), dtype=torch.float64, device=device)
  for k in range (0, CONST_K + 1):
    S = torch.float_power(S_keep, 2 ** (-k))

    aux = torch.matmul(torch.matmul(U,torch.diag(S)),Vt)

    aux = torch.t(torch.sqrt((torch.where(aux > TOL, aux, torch.tensor([TOL], device=device)))))
    X[k] = torch.t(torch.cat((aux, torch.reshape(torch.full((shortest_paths_mat.shape[0],), 2 ** (k * ALPHA - 2), device=device),(1, -1))), dim=0))

    del aux
    del S

  del U
  del Vt
  del S_keep

  return X







def padWithZeros_torch(X, left_margin, right_margin, top_margin, bottom_margin, dim=3):
    if dim == 3:
        newX = torch.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin, X.shape[2]), dtype=X.dtype, device=device)
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin, :] = X
    
    elif dim == 2:
        newX = torch.zeros((X.shape[0] + left_margin + right_margin, X.shape[1] + top_margin + bottom_margin), dtype=X.dtype, device=device)
        newX[left_margin:X.shape[0] + left_margin, top_margin:X.shape[1] + top_margin] = X

    else:
        newX = []

    return newX

def calc_patch_label_torch(labels, i, j, rows_factor, cols_factor, method='center'):
    if method=='center':
        return labels[i*rows_factor + rows_factor//2, j*cols_factor + cols_factor//2]
    elif method=='most_common':
        labels_patch = (labels[i*rows_factor : (i+1)*rows_factor, j*cols_factor : (j+1)*cols_factor]).int()
        counts = torch.bincount(labels_patch.flatten())

        # in order to not let 0 values take over and set many labels to 0 which leads to small number of non zero labeled patches
        counts[0]=1

        return torch.argmax(counts)
    
    print("ERROR- INCORRECT METHOD FOR LABELING PATCHES")

def patch_data_torch(data, labels, rows_factor, cols_factor, method_label_patch):
    rows, cols, channels = data.shape

    left_margin = ((-rows) % rows_factor) // 2
    right_margin = ((-rows) % rows_factor + 1) // 2
    top_margin = ((-cols) % cols_factor) // 2
    bottom_margin = ((-cols) % cols_factor + 1) // 2

    data = padWithZeros_torch(data, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin)
    labels = padWithZeros_torch(labels, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin, dim=2)


    new_rows, new_cols, _ = data.shape

    patched_data = torch.empty((new_rows // rows_factor, new_cols // cols_factor, rows_factor, cols_factor, channels), dtype=data.dtype, device=device)
    patched_labels = torch.zeros((patched_data.shape[0], patched_data.shape[1]), dtype=labels.dtype, device=device)

    for i in range(new_rows // rows_factor):
        for j in range(new_cols // cols_factor):
            datapoint = data[i*rows_factor: (i+1)*rows_factor, j*cols_factor: (j+1)*cols_factor, :]
            patched_data[i, j] = datapoint
            patched_labels[i, j] = calc_patch_label_torch(labels, i, j, rows_factor, cols_factor, method=method_label_patch)

    return patched_data, patched_labels, labels


def normalize_each_band_torch(X):
    
    X_normalized = torch.zeros_like(X, dtype=torch.float64, device=device)

    for i in range(X.shape[2]):
        X_band = X[:,:,i]
        scaled_data = torch.div((torch.sub(X_band,torch.min(X_band).item())),((torch.max(X_band)).item() - (torch.min(X_band)).item()))
        X_normalized[:,:,i] = scaled_data

    
    return X_normalized

import time

def calc_P_torch(d, apply_2_norm=False):
  epsilon = CONST_C*torch.median(d)
  W = torch.exp(-1*d/epsilon)

  if apply_2_norm:
    S_vec = torch.sum(W,dim=1)

    S = torch.diag(1/S_vec)
    
    W_gal = torch.matmul(torch.matmul(S,W),S)
    

    D_vec = torch.sum(W_gal,dim=1)
    D = torch.diag(1 / D_vec)
    P = torch.matmul(D,W_gal)

    del S_vec
    del S
    del W_gal
    del D_vec
    del D

  else:
    D_vec = torch.sum(W,dim=1)
    D = torch.diag(1/ D_vec)
    P = torch.matmul(D,W)

    del D_vec
    del D

  del W

  return P


def prepare_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    # print("$$$$$$$$$$ IN PREPERATION $$$$$$$$$$$$")

    # st = time.time()


    if is_normalize_each_band:
        X = normalize_each_band_torch(X)

    # print("NORMALIZATION: ", time.time()-st)
    # st = time.time()

    X_patches, y_patches, labels_padded= patch_data_torch(X, y, rows_factor, cols_factor, method_label_patch)
    
    # print("PATCHING: ", time.time()-st)
    # st = time.time()

    num_patches_in_row = y_patches.shape[1]

    y_patches = y_patches.flatten()


    X_patches = torch.reshape(X_patches, (-1, np.prod(X_patches.shape[2:])))

    # st = time.time()


    distances = torch.cdist(X_patches, X_patches)
    
    del X_patches
    # print("DISTANCES WITH CDIST: ", time.time()-st)
    # st = time.time()


    P = calc_P_torch(distances, apply_2_norm=True)

    # print("CALC_P: ", time.time()-st)


    return distances,P,y_patches,num_patches_in_row, labels_padded 









def calc_hdd_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    st = time.time()
    distances,P,y_patches,num_patches_in_row, labels_padded = prepare_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
    
    print("PREPARE TIME: ", time.time()-st)
    st = time.time()

    HDE = hde_torch(distances)

    del distances

    torch.abs(HDE, out=HDE)

    print("HDE TIME: ", time.time()-st)
    st = time.time()

    # print("HDE.shape: ", HDE.shape)
    
    hdd_mat = hdd_torch(HDE, P)

    del HDE
    # hdd_mat_2 = hdd(HDE, P)
    # print("NORM: ", np.linalg.norm(hdd_mat-hdd_mat_2))

    print("HDD TIME: ", time.time()-st)

    return hdd_mat, labels_padded, num_patches_in_row,y_patches



def whole_pipeline_all_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center'):
    print("XXXXXXX IN METHOD XXXXXXXXX")
    st = time.time()

    X = X.to(device)
    y = y.to(device)

    d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)

    print("WHOLE METHOD TIME: ", time.time()-st)
    st = time.time()

    print("XXXXXXX IN CLASSIFICATION XXXXXXXXX")
    n_neighbors = 3

    y_patches = y_patches.int()
    
    if torch.cuda.is_available():
        d_HDD = d_HDD.cpu()
        y_patches = y_patches.cpu()
        labels_padded = labels_padded.cpu()


    main(d_HDD.numpy(), y_patches.numpy(), n_neighbors, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row)

    print("WHOLE CLASSIFICATION TIME: ", time.time()-st)




def whole_pipeline_divided_torch(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center', is_print=False):
    st = time.time()
    
    num_patches = int(np.ceil(X.shape[0]/rows_factor)*np.ceil(X.shape[1]/cols_factor))

    distance_mat_arr = torch.zeros((X.shape[-1],num_patches,num_patches), device=device)
    for i in range(X.shape[-1]):
        if is_print:
            print((i+1)," out of: ", X.shape[-1])
        X_curr = torch.reshape(X[:,:,i], (X.shape[0],X.shape[1],1))
        d_HDD, labels_padded, num_patches_in_row,y_patches = calc_hdd_torch(X_curr,y, rows_factor, cols_factor, is_normalize_each_band=is_normalize_each_band, method_label_patch=method_label_patch)
        distance_mat_arr[i,:,:] = d_HDD

        if i!=X.shape[-1]-1:
            del X_curr
            del d_HDD
            del labels_padded
            del y_patches

    

    print("TOTAL TIME FOR METHOD: ", time.time()-st)

    n_neighbors = 3

    y_patches = y_patches.int()

    if torch.cuda.is_available():
        distance_mat_arr = distance_mat_arr.cpu()
        y_patches = y_patches.cpu()
        labels_padded = labels_padded.cpu()

    main_divided(distance_mat_arr.numpy(), y_patches.numpy(), n_neighbors, labels_padded.numpy(), rows_factor, cols_factor, num_patches_in_row)


from initial_plots import read_dataset

def test_torch_implementation():
    df = read_dataset(gt=False)

    X = np.array(df)
    X = X.reshape((610,340, 103))

    df = read_dataset(gt=True)
    y = np.array(df)

    rows_factor=21
    cols_factor=21

    distances,P,y_patches,num_patches_in_row, labels_padded = prepare(X,y, rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center')
    distances_c,P_c,y_patches_c,num_patches_in_row_c, labels_padded_c = prepare_torch(torch.from_numpy(X),torch.from_numpy(y), rows_factor, cols_factor, is_normalize_each_band=True, method_label_patch='center')

    print("NORM DISTANCES: ", np.linalg.norm(distances-(distances_c.numpy())))
    print("NORM P: ", np.linalg.norm(P-(P_c.numpy())))
    print("NORM y_patches: ", np.linalg.norm(y_patches-(y_patches_c.numpy())))
    print("NORM labels_padded: ", np.linalg.norm(labels_padded-(labels_padded_c.numpy())))
    print("DIFF num_patches_in_row: ", (num_patches_in_row-num_patches_in_row_c))

    HDE = hde(distances)
    HDE = np.abs(HDE)

    HDE_c = hde_torch(distances_c)
    HDE_c = torch.abs(HDE_c)

    print("NORM HDE: ", np.linalg.norm(HDE-(HDE_c.numpy())))
    
    hdd_mat = hdd_try(HDE, P)
    hdd_mat_C = hdd_torch(HDE_c, P_c)

    print("NORM HDD: ", np.linalg.norm(hdd_mat-(hdd_mat_C.numpy())))



# test_torch_implementation()