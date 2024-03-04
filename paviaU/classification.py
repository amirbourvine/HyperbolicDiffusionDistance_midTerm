import numpy as np
import random



class kNN():
    def __init__(self, n_neighbors:int = 3):
        # assume non-negative labels
        self.n_neighbors = n_neighbors

    def fit(self, labels, patch_to_points_dict):
      self.labels = labels
      self.patch_to_points_dict = patch_to_points_dict
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
            for i in range(i_start,i_end):
                for j in range(j_start,j_end):
                    if y[i,j]!=0:
                        total_preds += 1
                        if y[i,j] == predictions[ind]:
                            total_correct += 1

                        preds.append(predictions[ind])
                        gt.append(y[i,j])
        
        return total_correct/total_preds, preds,gt


def throw_0_labels(distances_mat, labels, patch_to_points_dict):
    non_zero_indices = np.where(labels != 0)[0]
    non_zero_indices = np.sort(non_zero_indices)

    new_res = {}
    for i in range(non_zero_indices.shape[0]):
        new_res[i] = patch_to_points_dict[non_zero_indices[i]]

    dmat_new = distances_mat[np.ix_(non_zero_indices, non_zero_indices)]
    labels_new = labels[np.ix_(non_zero_indices)]

    return dmat_new,labels_new, new_res

def split_train_test(distances_mat, labels, test_size = 0.2):
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
    dmat_train = distances_mat[np.ix_(indices_train, indices_train)]
    labels_train = labels[np.ix_(indices_train)]

    indices_test = []
    for i in range(labels.shape[0]):
        if i in indices_train:
            continue
        indices_test.append(i)
    indices_test = np.array(indices_test)
    # print(indices_test)

    dmat_test = distances_mat[np.ix_(indices_test, indices_train)]
    labels_test = labels[np.ix_(indices_test)]

    return indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test

def patch_to_points(labels, rows_factor, cols_factor, num_patches_in_row):
    """
    create a dict where key is i- index of patc in labels and value is (i_start, i_end, j_start, j_end)
    which are the boundaries of indices of points of this patch
    """
    res = {}
    for i in range(labels.shape[0]):
        i_patch = i // num_patches_in_row
        j_patch = i % num_patches_in_row

        i_start = i_patch*rows_factor
        j_start = j_patch*cols_factor
        res[i] = (i_start, i_start+rows_factor, j_start, j_start+cols_factor)
    
    return res


#TODO: find a way to calculate the indices of the points in each patch of the test, and then predict all of the points- each point by the label of its patch
def main(distances_mat, labels, n_neighbors, labels_padded, rows_factor, cols_factor, num_patches_in_row):
    patch_to_points_dict = patch_to_points(labels, rows_factor, cols_factor, num_patches_in_row)

    distances_mat, labels,patch_to_points_dict = throw_0_labels(distances_mat, labels,patch_to_points_dict)

    indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = split_train_test(distances_mat, labels, test_size = 0.2)

    clf = kNN(n_neighbors=n_neighbors)
    clf.fit(labels=labels_train, patch_to_points_dict=patch_to_points_dict)

    train_acc, train_preds,train_gt = clf.score(dmat_train, indices_train, labels_padded)
    test_acc, test_preds,test_gt= clf.score(dmat_test, indices_test, labels_padded)
    print("Train Accuracy: ",train_acc)
    print("Test Accuracy: ",test_acc)

    return train_acc,test_acc, test_preds,test_gt




    