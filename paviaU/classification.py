import numpy as np
import random
from sklearn.base import BaseEstimator,ClassifierMixin


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        # assume non-negative labels
        self.n_neighbors = n_neighbors

    def fit(self, labels):
      self.labels = labels
      return self

    def predict(self, X):
      distances = X

      y_mat = np.tile(self.labels,(distances.shape[0],1))

      sorted_distances_labels = np.take_along_axis(y_mat, np.argsort(distances, axis=1), axis=1)

      X_kNN = sorted_distances_labels[:, :self.n_neighbors]

      predictions = np.zeros(X_kNN.shape[0])
      for i in range(predictions.shape[0]):
          curr = np.bincount(X_kNN[i,:])
          predictions[i] = np.argmax(curr)

      return predictions


def throw_0_labels(distances_mat, labels):
    non_zero_indices = np.where(labels != 0)[0]
    non_zero_indices = np.sort(non_zero_indices)
    dmat_new = distances_mat[np.ix_(non_zero_indices, non_zero_indices)]
    labels_new = labels[np.ix_(non_zero_indices)]

    return dmat_new,labels_new

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
labels = np.zeros((10), dtype=int)

for i in range(10):
    for j in range(10):
        distances_mat[i,j] = 10*i+j
    labels[i] = i


print(distances_mat)
print(labels)
print("XXXXXXXXXXXXXXX")

distances_mat, labels = throw_0_labels(distances_mat, labels)

print(distances_mat)
print(labels)
print("XXXXXXXXXXXXXXX")

indices_train,dmat_train,labels_train,indices_test,dmat_test,labels_test = split_train_test(distances_mat, labels, test_size = 0.2)

print(dmat_train)
print(labels_train)
print(dmat_test)
print(labels_test) 

print("XXXXXXXXXXXXXXXXXXXXXX")
clf = kNN(n_neighbors=1)
clf.fit(labels=labels_train)
preds = clf.predict(dmat_test)
print(preds)
