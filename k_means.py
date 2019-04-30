import numpy as np
import matplotlib.pyplot as plt

# K-means Clustering
class KMeans():
    def __init__(self, k = 3, tol = 0.05, max_iter = 100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def initialize_centers(self, X_train):
        p = np.random.permutation(len(X_train))
        return X_train[p[:self.k]]

    def initialize_labels(self, X_train):
        len_c = len(X_train)//self.k
        rem_c = len(X_train)%self.k
        lab_train = [0]*len_c + [1]*len_c + [2]*len_c + [2]*rem_c
        np.random.shuffle(lab_train)
        return lab_train
        
    def assign_clusters(self, X_train, centers):
        ass = np.argmin(np.sqrt(np.sum(np.square(X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis = 2)), axis = 1)
        return ass

    def calculate_centers(self, X_train, ass):
        new_centers = np.zeros((self.k, 2))
        for i in np.unique(ass):
            new_centers[i, :] = np.mean(X_train[ass == i], axis = 0)
        return new_centers

    def predict(self, X_train, init_c = True):
        if init_c:
            c = self.initialize_centers(X_train)
        else:
            x = self.initialize_labels(X_train)
            c = self.calculate_centers(X_train, x)

        c_old = c
        for i in range(self.max_iter):
            x = self.assign_clusters(X_train, c)
            c = self.calculate_centers(X_train, x)
            if np.any(np.sqrt(np.sum(np.square(c-c_old),axis = 1)) < self.tol):
                print("Tolerance Reached")
                return x, c
            c_old = c
        return x, c

if __name__ == "__main__":

    # Create Dataset 3 clusters
    mean_1 = (0, 0)
    mean_2 = (3, 3)
    mean_3 = (-3, -3)
    cov = [[1, 0], [0, 1]]
    x_1 = np.random.multivariate_normal(mean_1, cov, 100)
    x_2 = np.random.multivariate_normal(mean_2, cov, 100)
    x_3 = np.random.multivariate_normal(mean_3, cov, 100)

    X_train = np.r_[x_1, x_2, x_3]

    clf = KMeans(3, 0.05, 100)
    lab, clus = clf.predict(X_train, init_c = True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    ax1.scatter(X_train[:,0], X_train[:,1])
    ax2.scatter(X_train[:, 0], X_train[:,1], c = lab)
    plt.show()

