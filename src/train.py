
from typing import Any, List, Tuple
from matplotlib import pyplot as plt
from mne import Epochs, concatenate_raws, pick_types, set_log_level
from mne.channels import make_standard_montage, read_custom_montage
from mne.decoding import CSP, Scaler
from mne.io import BaseRaw, read_raw_edf
import numpy as np
from pydantic import ConfigDict, validate_call
from mne.datasets.eegbci import standardize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from csp import CSP42
from plot import make_exclude_list
from utils import load_data

experiments = [
	[3, 7, 11],
	[4, 8, 12],
	[5, 9, 13],
	[6, 10, 14],
	[3, 5, 7, 9, 11, 13],
	[4, 6, 8, 10, 12, 14],
]

model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call(config=model_config)
def split_datasets(epochs: Epochs) -> Tuple[np.ndarray, np.ndarray, Any, np.ndarray, np.ndarray]:
	data = epochs.get_data(copy=False)
	Y = epochs.events[:, -1] - 2

	splitter = ShuffleSplit(1, test_size=0.2)

	(X_data, X_test) = next(splitter.split(data))

	folder = KFold()

	X = data[X_data]
	Y_data = Y[X_data]
	Y_test = Y[X_test]

	return (Y_data, X, folder.split(X_data), X_test, Y_test)

def train():
	global total_correct, total_states
	set_log_level(False)

	accuracies = []

	for subject in range(1, 40):
		# csp = CSP(n_components=8, reg=None, log=False, norm_trace=False)
		epochs = load_data([subject], [i for i in range(3, 15)])


		scaler = Scaler(epochs.info)
		csp = CSP42(n_components=4)
		model = LinearDiscriminantAnalysis(solver='lsqr')

		pipe = Pipeline([("Scaler", scaler), ("CSP", csp), ("EST", model)])

		X = epochs.get_data(copy=False).astype(np.float64)
		Y = epochs.events[:, -1] - 1

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		pipe.fit(X_train, Y_train)

		for i, experiment in enumerate(experiments):
			print(f"-- Subject #{subject}, Experiment #{i} ({experiment})")

			epochs = load_data([subject], [i for i in range(3, 15)])
			X = epochs.get_data(copy=False).astype(np.float64)
			Y = epochs.events[:, -1] - 1

			X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

			acc = pipe.score(X_test, Y_test)

			print(f"Test Accuracy: {acc * 100:0.2f}%")

			accuracies.append(acc)

			# scores = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5, shuffle = True))
			# print(f"{100 * scores.mean():0.2f}% accuracy with a standard deviation of {100 * scores.std():0.2f}%" )
		
		
		print(f"======== Total accuracy: {sum(accuracies) / len(accuracies) * 100:0.2f}%")





