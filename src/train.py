
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

		new_raws = [
			read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True) for run in runs
		]


		# print([r.info['sfreq'] for r in new_raws])

		raws += new_raws

	raw = concatenate_raws(raws)

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
	raw.filter(7.0, 32.0, 
			fir_design="firwin", 
			)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		proj=True,
		event_id=["hands", "feet"],
		picks=picks,
		baseline=None,
		# detrend=1,
		# tmin=1,
		# tmax=2,
		preload=True,
	)

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

	for subject in range(1, 110):
		csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
		model = LinearDiscriminantAnalysis(solver='lsqr')

		pipe = Pipeline([("CSP", csp), ("EST", model)])

		epochs = load_data([subject], [i for i in range(3, 15)])

		X = epochs.get_data(copy=False).astype(np.float64)
		Y = epochs.events[:, -1] - 2

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		pipe.fit(X_train, Y_train)

		for i, experiment in enumerate(experiments):
			print(f"-- Subject #{subject}, Experiment #{i} ({experiment})")

			epochs = load_data([subject], [i for i in range(3, 15)])
			X = epochs.get_data(copy=False).astype(np.float64)
			Y = epochs.events[:, -1] - 2

			X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

			acc = pipe.score(X_test, Y_test)

			print(f"Test Accuracy: {acc * 100:0.2f}%")

			scores = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5, shuffle = True))
			print(f"{100 * scores.mean():0.2f}% accuracy with a standard deviation of {100 * scores.std():0.2f}%" )




