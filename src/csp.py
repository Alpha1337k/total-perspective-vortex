
from matplotlib import pyplot as plt
from mne.decoding import CSP
import numpy as np
from numpy import linalg
from train import load_data

def test_real():
	csp = CSP(n_components=4)
	epochs = load_data([1], [i for i in range(1,14)])

	X = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 1

	print(Y[0:1])

	csp.fit(X, Y)

	X1Res = csp.transform(X[0:1])
	X2Res = csp.transform(X[1:2])
	
	plt.scatter(X1Res[0][0], X1Res[0][1], c='blue')
	plt.scatter(X2Res[0][0], X2Res[0][1], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()

def run_csp():
# 	test_real()
# # 
# 	exit(1)

	epochs = load_data([1], [i for i in range(1,15)])

	components = 8

	X = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 1

	print(X.shape)

	# X = X.mean(axis=1)
	# print(X.shape)

	print(Y)

	X1: np.ndarray = X[np.where(Y == 0)]
	X2: np.ndarray = X[np.where(Y == 1)]

	plt.scatter(X1[:, 0], X1[:, 2], c='red')
	plt.scatter(X2[:, 0], X2[:, 2], c='blue')
	plt.savefig("results/csp/before.png")
	plt.clf()

	print("Divided: ", X1.shape, X2.shape)

	# R1 = (X1 @ X1.transpose()) / len(X1)
	# R2 = (X2 @ X2.transpose()) / len(X2)

	print("TPS: ", X1.shape)

	R1 = np.cov(X1.transpose(1,0,2).reshape(40, -1))
	R2 = np.cov(X2.transpose(1,0,2).reshape(40, -1))
	

	print(R1.shape)
	print(R2.shape)

	eigen_values, eigen_vectors = linalg.eig(R1)

	sorted_index = np.argsort(eigen_values)

	eigen_vectors = eigen_vectors[:, sorted_index]

	print("----\n\n")

	filter = eigen_vectors[ : components]

	print("Filter size: ", filter.shape)
	print("X1 size: ", X1.shape)

	XRes = np.asarray([filter @ x for x in X]).mean(axis=2)

	XRes = XRes / np.linalg.norm(XRes)

	X1Res: np.ndarray = XRes[np.where(Y == 0)]
	X2Res: np.ndarray = XRes[np.where(Y == 1)]	

	print(X1Res.shape)
	print("----\n\n")
	print(X2Res.shape)


	plt.scatter(X1Res[:, 0], X1Res[:, 5], c='blue')
	plt.scatter(X2Res[:, 0], X2Res[:, 5], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()