
from typing import Any, List, Tuple
from matplotlib import pyplot as plt
from mne import Epochs, concatenate_raws, pick_types, set_log_level
from mne.channels import make_standard_montage, read_custom_montage
from mne.decoding import CSP, Scaler
from mne.io import BaseRaw, read_raw_edf
import numpy as np
from pydantic import ConfigDict, validate_call
from mne.datasets.eegbci import standardize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from plot import make_exclude_list

experiments = [
	[3, 7, 11],
	[4, 8, 12],
	[5, 9, 13],
	[6, 10, 14],
	[3, 5, 7, 9, 11, 13],
	[4, 6, 8, 10, 12, 14],
]

model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call
def load_data(sessions: List[int], runs: List[int]):
	raws = []

	for session in sessions:
		# print(f"LOADING session #{session} ({len(raws)})")

		if (session in [88, 92, 100]):
			print(f"skipping #{session}")
			continue

		new_raws = [
		read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True) for run in runs
		]

		# print([r.info['sfreq'] for r in new_raws])

		raws += new_raws
		
	print("Merging..")

	raw = concatenate_raws(raws)

	print("Done!")

	raws = None

	standardize(raw)  # set channel names
	# montage = make_standard_montage("standard_1005")
	montage = read_custom_montage("./data/custom_fixture.txt")

	excluded_channels = make_exclude_list(montage.ch_names)

	# print(excluded_channels)

	raw.drop_channels(excluded_channels)

	raw.set_montage(montage)

	raw.annotations.rename(dict(T0="rest", T1="hands", T2="feet"))  # as documented on PhysioNet

	raw.set_eeg_reference(projection=True)

	raw.notch_filter(60, method="iir")
	raw.filter(7.0, 30.0, 
			fir_design="firwin", 
			)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	print("Split to epochs..")

	epochs = Epochs(
		raw,
		proj=False,
		event_id=["hands", "feet"],
		picks=picks,
		baseline=None,
		# tmin=0,
		# tmax=1,
		# preload=True,
	)

	print("Done!")

	return epochs


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

@validate_call(config=model_config)
def fit(model: Pipeline, X: np.ndarray, Y: np.ndarray): 

	model.fit(data, labels)

	return model

def train():
	global total_correct, total_states
	set_log_level(False)

	csp = CSP(n_components=6, reg=None, log=True, norm_trace=True)
	model = LinearDiscriminantAnalysis()
	scaler = Scaler(scalings="mean")

	pipe = Pipeline([("SCL", scaler), ("CSP", csp), ("EST", model)])

	for i, experiment in enumerate(experiments):
		print(f"[] [] [] [] Experiment #{i} ({experiment})")

		epochs = load_data([i for i in range(1, 80)], experiment)

		X = epochs.get_data(copy=False).astype(np.float64)
		Y = epochs.events[:, -1] - 2

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		print(len(X_train), X_train[0].shape, Y_train[0:2])

		csp.fit(X_train, Y_train)

		# locations = [
		# 	np.where(Y_train == -1)[0],
		# 	np.where(Y_train == 0)[0],
		# 	# np.where(Y_train == 1)[0],
		# ]

		# T_0 = X_train[locations[0]] #csp.transform(X_train[locations[0]])  
		# T_1 = X_train[locations[1]] #csp.transform(X_train[locations[1]])  
		# # T_2 = csp.transform(X_train[locations[2]])

		# for n1 in range(0, 8):
		# 	for n2 in range(n1 + 1, 8):
		# 		plt.scatter(x=T_0[:, n1], y=T_0[:, n2], c='red', alpha=0.1)
		# 		plt.scatter(x=T_1[:, n1], y=T_1[:, n2], c='blue', alpha=0.1)
		# 		plt.savefig(f"./results/scatter-csp-{n1}-{n2}.png")
		# 		plt.clf()

		# exit(1)

		# csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)

		# transformed = csp.fit_transform(X_train, Y_train)

		# csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

		# plt.savefig("./results/test.png")

		# plt.clf()

		# model.partial_fit(transformed, Y_train, classes=[0,1,-1])

		print("START CV")

		# scores = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5, shuffle = True))

		# print(scores)
		# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

		pipe.fit(X_train, Y_train)


		acc = pipe.score(X_test, Y_test)

		print(acc, pipe.score(X_train, Y_train))

	# pipe.fit(X, Y)
	# csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)

	# Y_pred = pipe.predict(X_test)

	# print("__--__--DONE--__--__")

	# print(f"--- RESULTS ---")
	# print(f"{100 * accuracy_score(Y_test, Y_pred):.2f}% correct.")
	# print(f"{accuracy_score(Y_test, Y_pred, normalize=False):.2f} out of {len(Y_test)} correct.")
	# print(f"--- RESULTS ---")



