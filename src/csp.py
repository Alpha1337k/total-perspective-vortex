
from matplotlib import pyplot as plt
from mne.decoding import CSP, Scaler
from mne import set_log_level
import numpy as np
from scipy import linalg
import sklearn
import sklearn.preprocessing
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

		R1 = np.cov(X1.transpose(1,0,2).reshape(X.shape[1], -1))
		R2 = np.cov(X2.transpose(1,0,2).reshape(X.shape[1], -1))

		def epoch_cov(X):
			XData = X.transpose(1,0,2).reshape(X.shape[1], -1)
			return np.cov(XData)

		R1 = epoch_cov(X1)
		R2 = epoch_cov(X2)

		combined = np.asarray([R1, R2]).sum(axis=0) + np.eye(R1.shape[0]) * 1e-5

		tries = 0
		mults = [1e-12, 1e-11, 1e-10, 1e-8]

		while tries < len(mults):
			try:
				eigen_values, eigen_vectors = linalg.eigh(R1, combined)
				break
			except linalg.LinAlgError:
				# print("--- CORRECTING ---")
				combined = (R1 + R2) + np.eye(R1.shape[0]) * mults[tries]
				tries += 1

		if (tries == len(mults)):
			raise Exception("Could not find closure for non positive definite B")

		sorted_index = np.argsort(np.abs(eigen_values - 0.5))[::-1]
		eigen_vectors = eigen_vectors[:, sorted_index]

		self.filter = eigen_vectors.T[: self.n_components]

		# print(self.filter.shape, self.filter.mean(axis=1))
		XRes = np.asarray([self.filter @ x for x in X])

		self.normalize(XRes, True)

		return self
	
	def normalize(self, X, train: bool):
		X = np.mean(X ** 2, axis=2)

		# if (train):
		# 	self.mean = X.mean()
		# 	self.std  = X.std()


		# X /= self.std
		# X -= self.mean

		return X

	def transform(self, X):		
		XRes = np.asarray([self.filter @ x for x in X])
		
		return self.normalize(XRes, train=False)

def run_csp():
	set_log_level(False)
	epochs = load_data([1], [i for i in range(3,15)])

	n_components=6

	csp = CSP42(n_components=n_components)

	X = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 1

	print(X.shape)

	# X = X.mean(axis=1)
	# print(X.shape)

	print(Y)

	X1: np.ndarray = X[np.where(Y == 0)]
	X2: np.ndarray = X[np.where(Y == 1)]


	XRes = csp.fit_transform(X, Y)


	X1Res: np.ndarray = XRes[np.where(Y == 0)]
	X2Res: np.ndarray = XRes[np.where(Y == 1)]


	for i in range(n_components):
		for n in range(i + 1, n_components):
			plt.scatter(X1Res[:, i], X1Res[:, n], c='red', alpha=0.1)
			plt.scatter(X2Res[:, i], X2Res[:, n], c='blue', alpha=0.1)
			plt.savefig(f"results/csp/{i}-{n}.png")
			plt.clf()
