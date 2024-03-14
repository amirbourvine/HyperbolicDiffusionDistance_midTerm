import numpy as np
import random
import time



class kNN():
    def __init__(self, n_neighbors:int = 3, is_divided = False):
        # assume non-negative labels
        self.n_neighbors = n_neighbors
        self.is_divided = is_divided

    def fit(self, labels, patch_to_points_dict):
      self.labels = labels
      self.patch_to_points_dict = patch_to_points_dict

      print("patch_to_points_dict: ", patch_to_points_dict)
      print("labels: ", labels)
      return self

    def score(self, distances, indices_test, y):
        y_mat = np.tile(self.labels,(distances.shape[0],1))

        sorted_distances_labels = np.take_along_axis(y_mat, np.argsort(distances, axis=1), axis=1)

        X_kNN = sorted_distances_labels[:, :self.n_neighbors]

        predictions = np.zeros(X_kNN.shape[0])
        for i in range(predictions.shape[0]):
            curr = np.bincount(X_kNN[i,:])
            predictions[i] = np.argmax(curr)


        print("Number of patches: ", predictions.shape[0])

        total_preds = 0
        total_correct = 0
        preds = []
        gt = []
        for ind in range(predictions.shape[0]):
            ind_patch = indices_test[ind]
            i_start,i_end,j_start,j_end = self.patch_to_points_dict[ind_patch]
            i = (i_start + i_end) // 2
            j = (j_start + j_end) // 2

            if y[i,j]!=0:
                total_preds += 1
                if y[i,j] == predictions[ind]:
                    total_correct += 1

                preds.append(predictions[ind])
                gt.append(y[i,j])


            # for i in range(i_start,i_end):
            #     for j in range(j_start,j_end):
            #         if y[i,j]!=0:
            #             total_preds += 1
            #             if y[i,j] == predictions[ind]:
            #                 total_correct += 1

            #             preds.append(predictions[ind])
            #             gt.append(y[i,j])
        
        return total_correct/total_preds, preds,gt
    
    def score_divided(self, distances_arr, indices_test, y):
        y_mat = np.tile(self.labels,(distances_arr[0].shape[0],1))

        sorted_distances_labels = np.ndarray(shape=(distances_arr.shape[0],distances_arr[0].shape[0],distances_arr[0].shape[1]))
        for i in range(distances_arr.shape[0]):
            sorted_distances_labels[i,:,:] = np.take_along_axis(y_mat, np.argsort(distances_arr[i], axis=1), axis=1)

        X_kNN = sorted_distances_labels[:,:, :self.n_neighbors]

        X_kNN = np.swapaxes(X_kNN,0,1)
        X_kNN = X_kNN.reshape((X_kNN.shape[0],X_kNN.shape[1]*X_kNN.shape[2]))
        X_kNN = (X_kNN).astype(int)
        print("X_kNN.shape: ", X_kNN.shape)

        predictions = np.zeros(X_kNN.shape[0])
        for i in range(predictions.shape[0]):
            curr = np.bincount(X_kNN[i,:])
            predictions[i] = np.argmax(curr)


        print("Number of patches: ", predictions.shape[0])

        total_preds = 0
        total_correct = 0
        preds = []
        gt = []
        for ind in range(predictions.shape[0]):
            ind_patch = indices_test[ind]
            i_start,i_end,j_start,j_end = self.patch_to_points_dict[ind_patch]
            for i in range(i_start,i_end):
                for j in range(j_start,j_end):
                    if y[i,j]!=0:
                        total_preds += 1
                        if y[i,j] == predictions[ind]:
                            total_correct += 1

                        preds.append(predictions[ind])
                        gt.append(y[i,j])
        
        return total_correct/total_preds, preds,gt


def throw_0_labels(distances_mat, labels, patch_to_points_dict, is_divided=False):
    non_zero_indices = np.where(labels != 0)[0]
    non_zero_indices = np.sort(non_zero_indices)

    new_res = {}
    for i in range(non_zero_indices.shape[0]):
        new_res[i] = patch_to_points_dict[non_zero_indices[i]]

    labels_new = labels[np.ix_(non_zero_indices)]

    if not is_divided:
        dmat_new = distances_mat[np.ix_(non_zero_indices, non_zero_indices)]
        return dmat_new,labels_new, new_res
    else:
        distances_mat_new = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
        for i in range(distances_mat_new.shape[0]):
            distances_mat_new[i] = (distances_mat[i])[np.ix_(non_zero_indices, non_zero_indices)]

        return distances_mat_new,labels_new, new_res


    

    

def split_train_test(distances_mat, labels, test_size = 0.2, is_divided=False):
    """
    labels is np.array
    returns the train-test split:
    for train- labels and distances is train_numXtrain_num mat- distances between all the training points
    for test- distances is test_numXtrain_num mat- distances between each test point to all of the train points
    """
    num_training = int(labels.shape[0]*(1-test_size))
    indices_train = random.sample(range(0, labels.shape[0]), num_training)
    indices_train = np.sort(indices_train)
    # print(indices_train)
    if not is_divided:
        dmat_train = distances_mat[np.ix_(indices_train, indices_train)]
    else:
        dmat_train = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
        for i in range(dmat_train.shape[0]):
            dmat_train[i] = (distances_mat[i])[np.ix_(indices_train, indices_train)]
    
    labels_train = labels[np.ix_(indices_train)]

    indices_test = []
    for i in range(labels.shape[0]):
        if i in indices_train:
            continue
        indices_test.append(i)
    
    indices_test = np.array(indices_test)
    # print(indices_test)

    if not is_divided:
        dmat_test = distances_mat[np.ix_(indices_test, indices_train)]
    else:
        dmat_test = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
        for i in range(dmat_test.shape[0]):
            dmat_test[i] = (distances_mat[i])[np.ix_(indices_test, indices_train)]

    labels_test = labels[np.ix_(indices_test)]

    return indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test

# def split_train_test(point_to_patches, labels, test_size = 0.2, is_divided=False):
#     rows_size = max([t[1] for t in point_to_patches.keys()]) + 1
    
#     num_training = int(len(point_to_patches.keys())*(1-test_size))
#     points_train = random.sample(range(0, len(point_to_patches.keys()) - 1), num_training)
#     points_train = np.sort(points_train)
#     points_train = [(rand // rows_size, rand % rows_size) for rand in points_train]

#     # print(indices_train)
#     # if not is_divided:
#     #     dmat_train = distances_mat[np.ix_(indices_train, indices_train)]
#     # else:
#     #     dmat_train = np.ndarray(shape=(distances_mat.shape[0],), dtype=np.ndarray)
#     #     for i in range(dmat_train.shape[0]):
#     #         dmat_train[i] = (distances_mat[i])[np.ix_(indices_train, indices_train)]
    
#     def most_common_element(lst):
#         unique_elements, counts = np.unique(lst, return_counts=True)
#         max_count_index = np.argmax(counts)
#         return unique_elements[max_count_index]
    
#     labels_train = {}
#     labels_test = {}
#     for point in point_to_patches.keys():
#         patches_indices = point_to_patches[point]
#         patches_labels = [labels[ind] for ind in patches_indices]

#         if point in points_train:
#             labels_train[point] = most_common_element(patches_labels) #internet trick for most common element

#         else:
#             labels_test[point] = most_common_element(patches_labels) #internet trick for most common element


#     return points_train, labels_train, labels_test.keys(), labels_test



def patch_to_points(labels, rows_factor, cols_factor, rows_overlap, cols_overlap, num_patches_in_row):
    """
    create a dict where key is i- index of patch in labels and value is (i_start, i_end, j_start, j_end)
    which are the boundaries of indices of points of this patch
    """
    if(rows_overlap == -1 or cols_overlap==-1):
        res = {}
        for i in range(labels.shape[0]):
            i_patch = i // num_patches_in_row
            j_patch = i % num_patches_in_row

            i_start = i_patch*rows_factor
            j_start = j_patch*cols_factor
            res[i] = (i_start, i_start+rows_factor, j_start, j_start+cols_factor)
        
        return res

    #Same calculation but with overlapping patches:
    res = {}
    for i in range(labels.shape[0]):
        i_patch = i // num_patches_in_row
        j_patch = i % num_patches_in_row

        i_start = i_patch*rows_overlap
        j_start = j_patch*cols_overlap
        res[i] = (i_start, i_start+rows_factor, j_start, j_start+cols_factor)
    
    return res

def point_to_patches(patch_to_points):
    res = {}
    for patch_ind, patch_range in patch_to_points.items():
        for i in range(patch_range[0], patch_range[1]):
            for j in range(patch_range[2], patch_range[3]):
                if (i, j) in res:
                    res[(i, j)].append(patch_ind)
                else:
                    res[(i, j)] = [patch_ind]

    return res

def main(distances_mat, labels, n_neighbors, labels_padded, rows_factor, cols_factor, rows_overlap, cols_overlap, num_patches_in_row):

    st = time.time()

    patch_to_points_dict = patch_to_points(labels, rows_factor, cols_factor,  rows_overlap, cols_overlap, num_patches_in_row)

    distances_mat, labels,patch_to_points_dict = throw_0_labels(distances_mat, labels,patch_to_points_dict)

    indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = split_train_test(distances_mat, labels, test_size = 0.2)

    print("DICT CREATION, THROW 0 LABELS, SPLIT TETS TRAIN TIME: ", time.time()-st)

    clf = kNN(n_neighbors=n_neighbors)

    clf.fit(labels=labels_train, patch_to_points_dict=patch_to_points_dict)  

    st = time.time()

    train_acc, train_preds,train_gt = clf.score(dmat_train, indices_train, labels_padded)
    test_acc, test_preds,test_gt= clf.score(dmat_test, indices_test, labels_padded)
    print("Train Accuracy: ",train_acc)
    print("Test Accuracy: ",test_acc)

    print("SCORES TIME: ", time.time()-st)

    return train_acc,test_acc, test_preds,test_gt

def main_divided(distances_mat_arr, labels, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row):
    patch_to_points_dict = patch_to_points(labels, rows_factor, cols_factor, num_patches_in_row)

    distances_mat_arr, labels,patch_to_points_dict = throw_0_labels(distances_mat_arr, labels,patch_to_points_dict, is_divided=True)

    indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = split_train_test(distances_mat_arr, labels, test_size = 0.2, is_divided=True)

    clf = kNN(n_neighbors=n_neighbors, is_divided = True)
    clf.fit(labels=labels_train, patch_to_points_dict=patch_to_points_dict)

    train_acc, train_preds,train_gt = clf.score_divided(dmat_train, indices_train, labels_padded)
    test_acc, test_preds,test_gt= clf.score_divided(dmat_test, indices_test, labels_padded)
    print("Train Accuracy: ",train_acc)
    print("Test Accuracy: ",test_acc)

    return train_acc,test_acc, test_preds,test_gt
