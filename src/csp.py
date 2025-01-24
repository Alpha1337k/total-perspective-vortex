
from matplotlib import pyplot as plt
from mne.decoding import CSP
from mne import set_log_level
import numpy as np
from scipy import linalg
from utils import load_data
from sklearn.base import BaseEstimator, TransformerMixin

class CSP42(TransformerMixin, BaseEstimator):
	filter = None
	normalizer = None
	mean = 0.0
	std = 0.0

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

		def epoch_cov(X):
			XData = X.transpose(1,0,2).reshape(40, -1)
			return np.cov(XData)

		R1 = epoch_cov(X1)
		R2 = epoch_cov(X2)

		combined = (R1 + R2) + np.eye(R1.shape[0]) * 1e-13

		tries = 0
		mults = [1e-11, 1e-10, 1e-8]

		while tries < 3:
			try:
				eigen_values, eigen_vectors = linalg.eigh(R1, combined)
				break
			except linalg.LinAlgError:
				print("--- CORRECTING ---")
				combined = (R1 + R2) + np.eye(R1.shape[0]) * mults[tries]
				tries += 1

		if (tries == 3):
			raise Exception("Could not find closure for non positive definite B")

		sorted_index = np.argsort(np.abs(eigen_values - 0.5))[::-1]
		eigen_vectors = eigen_vectors[:, sorted_index]

		self.filter = eigen_vectors.T[: self.n_components]

		print(self.filter.shape, self.filter.mean(axis=1))

		self.normalizer = np.linalg.norm(X1)

		X = (X ** 2).mean(axis=2)

		self.mean = X.mean()
		self.std  = X.std()

		return self

	def transform(self, X):		
		XRes = np.asarray([self.filter @ x for x in X])


		# assert(XRes.shape[1] == self.n_components)
		# XRes = np.log(XRes)

		# norm = np.linalg.norm(XRes)

		# XRes /= norm
		# np.nan_to_num(XRes, False, posinf=0.0, neginf=0.0)

		XRes = (XRes ** 2).mean(axis=2)


		XRes -= self.mean
		XRes /= self.std
		
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
	set_log_level(False)
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

	csp = CSP42(n_components=4)

	XRes = csp.fit_transform(X, Y)

	X1Res: np.ndarray = XRes[np.where(Y == 0)]
	X2Res: np.ndarray = XRes[np.where(Y == 1)]

	plt.scatter(X1Res[:, 1], X1Res[:, 3], c='blue')
	plt.scatter(X2Res[:, 1], X2Res[:, 3], c='red')
	plt.savefig("results/csp/after.png")
	plt.clf()