
from matplotlib import pyplot as plt
from mne.decoding import CSP
import numpy as np
from numpy import linalg
from train import load_data

def test_real():
	csp = CSP(n_components=4)
	epochs = load_data([1], [3])

	X = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 1

	print(Y[0:1])

	csp.fit(X[0:2], Y[0:2])

	X1Res = csp.transform(X[0:1])
	X2Res = csp.transform(X[1:2])
	
	plt.scatter(X1Res[0][0], X1Res[0][1], c='blue')
	plt.scatter(X2Res[0][0], X2Res[0][1], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()

def run_csp():
	# test_real()
# 
	# exit(1)

	epochs = load_data([1], [3])

	components = 4

	X = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 1

	print(X.shape)

	X = X.mean(axis=1)

	print(X.shape)

	print(Y)

	X1: np.ndarray = X[np.where(Y[:-1] == 0)]
	X2: np.ndarray = X[np.where(Y[:-1] == 1)]

	plt.scatter(X1[:, 0], X1[:, 1], c='red')
	plt.scatter(X2[:, 0], X2[:, 1], c='blue')
	plt.savefig("results/csp/before.png")
	plt.clf()

	print("Divided: ", X1.shape, X2.shape)

	# R1 = (X1 @ X1.transpose()) / len(X1)
	# R2 = (X2 @ X2.transpose()) / len(X2)

	R1 = np.cov(X1)
	R2 = np.cov(X2)
	

	print(R1.shape)
	print(R2.shape)

	eigen_values, eigen_vectors = linalg.eig(R1)

	sorted_index = np.argsort(eigen_values)

	eigen_vectors = eigen_vectors[:, sorted_index]

	print("----\n\n")

	filter = eigen_vectors[ : components]

	X1Res = (filter @ X1).transpose()
	X2Res = (filter @ X2).transpose()

	print(X1Res.shape)
	print("----\n\n")
	print(X2Res.shape)


	plt.scatter(X1Res[:, 0], X1Res[:, 2], c='blue')
	plt.scatter(X2Res[:, 0], X2Res[:, 2], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()