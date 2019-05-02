import numpy as np
import matplotlib.pyplot as plt

class kNN():
	def __init__(self, k = 5):
		self.k = k
	def predict(self, x, X_train, y_train):
		if len(x.shape) == 1:
			x = x[np.newaxis, :]
		'''Classify a new point according to KNN
		----------
		Inputs
		x_new: new datapoint to classify
		X_train: dataset of labelled points
		y_train: labels of dataset
		k: Number of nearest neighbouts to select for voting

		Output
		Class label of the new datapoint'''

#        dist_arr = np.sqrt(np.sum(np.square(x[np.newaxis, :] - X_train), axis = 1))
		dist_arr = np.sqrt(np.sum(x**2, axis = 1).reshape(x.shape[0], 1) + np.sum(X_train**2, axis = 1) - 2*np.matmul(x, X_train.T))
		y_pred = []
		for i in range(len(x)):
		    arr_sel = np.argsort(dist_arr[i])[:self.k] # argsort is from lowest to highest
		    unique, counts = np.unique(y_train[arr_sel], return_counts=True)

		    y_pred.append(unique[np.argmax(counts)])

		return y_pred

if __name__=="__main__":
	# Create Dataset 3 clusters
	mean_1 = (0, 0)
	mean_2 = (3, 3)
	mean_3 = (-3, -3)
	cov = [[1, 0], [0, 1]]
	x_1 = np.random.multivariate_normal(mean_1, cov, 100)
	x_2 = np.random.multivariate_normal(mean_2, cov, 100)
	x_3 = np.random.multivariate_normal(mean_3, cov, 100)

	X_train = np.r_[x_1, x_2, x_3]
	y_train = np.array([0]*len(x_1) + [1]*len(x_2) + [2]*len(x_3))

	cls = kNN(5)

	len_test = 10000
	x = np.r_[np.random.multivariate_normal(mean_1, cov, len_test), np.random.multivariate_normal(mean_2, cov, len_test),np.random.multivariate_normal(mean_3, cov, len_test)]
	y = np.array([0]*len_test + [1]*len_test + [2]*len_test)

	lab = cls.predict(x, X_train, y_train)

	print("Accuracy: {0:.2f}%".format(sum(y == lab)*100/len(lab)))

	# print("New Data Point: {}".format(x))
	# print("Class Label: {}".format(lab))
