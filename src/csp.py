
from matplotlib import pyplot as plt
# from mne.decoding import CSP
import numpy as np
from numpy import linalg
from utils import load_data
from sklearn.base import BaseEstimator, TransformerMixin

class CSP42(TransformerMixin, BaseEstimator):
	filter = None
	normalizer = None

	def __init__(self, *, n_components=4):
		self.n_components = n_components
	def fit(self, X, y):
		X1: np.ndarray = X[np.where(y == 0)]
		X2: np.ndarray = X[np.where(y == 1)]

		min_len = min(len(X1), len(X2))

		X1 = X1[:min_len - 1]
		X2 = X2[:min_len - 1]

		R1 = np.cov(X1.transpose(1,0,2).reshape(40, -1))
		R2 = np.cov(X2.transpose(1,0,2).reshape(40, -1))

		combined = (R1 + R2) ** 2 @ R1

		eigen_values, eigen_vectors = linalg.eigh(combined)

		sorted_index = np.argsort(eigen_values)
		eigen_vectors = eigen_vectors[:, sorted_index]

		self.filter = eigen_vectors[ : self.n_components]
		self.normalizer = np.linalg.norm(X1)

		return self

	def transform(self, X):		
		XRes = np.asarray([self.filter @ x for x in X]).mean(axis=2)

		# XRes = np.log(XRes)

		# norm = np.linalg.norm(XRes)

		# XRes /= norm
		# np.nan_to_num(XRes, False, posinf=0.0, neginf=0.0)
		
		return XRes


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

	csp = CSP42(n_components=8)

	XRes = csp.fit_transform(X, Y)

	X1Res: np.ndarray = XRes[np.where(Y == 0)]
	X2Res: np.ndarray = XRes[np.where(Y == 1)]

	plt.scatter(X1Res[:, 1], X1Res[:, 4], c='blue')
	plt.scatter(X2Res[:, 1], X2Res[:, 4], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()