import numpy as np
import random

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




distances_mat = np.zeros((10,10))
labels = np.zeros((10))

for i in range(10):
    for j in range(10):
        distances_mat[i,j] = i*10+j
    labels[i] = i

# print(distances_mat)
# print("XXXXXXXXXXXXXX")
# print(labels)
    

indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = split_train_test(distances_mat, labels, test_size = 0.2)

print(dmat_train)
print(labels_train)
print(dmat_test)
print(labels_test)  