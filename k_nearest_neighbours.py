import numpy as np
import matplotlib.pyplot as plt

class kNN():
	def __init__(self, k = 5):
		self.k = k
	def predict(self, x, X_train, y_train):
		'''Classify a new point according to KNN
		----------
		Inputs
		x_new: new datapoint to classify
		X_train: dataset of labelled points
		y_train: labels of dataset
		k: Number of nearest neighbouts to select for voting

		Output
		Class label of the new datapoint'''

		dist_arr = np.sqrt(np.sum(np.square(x[np.newaxis, :] - X_train), axis = 1))
		arr_sel = np.argsort(dist_arr)[:self.k] # argsort is from lowest to highest
		unique, counts = np.unique(y_train[arr_sel], return_counts=True)

		return np.argmax(counts)

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

	x = np.array([0,0])
	lab = cls.predict(x, X_train, y_train)

	print("New Data Point: {}".format(x))
	print("Class Label: {}".format(lab))