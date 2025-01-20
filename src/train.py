
from typing import Any, List, Tuple
from mne import Epochs, concatenate_raws, pick_types, set_log_level
from mne.channels import make_standard_montage, read_custom_montage
from mne.decoding import CSP
from mne.io import BaseRaw, read_raw_edf
import numpy as np
from pydantic import ConfigDict, validate_call
from mne.datasets.eegbci import standardize
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from plot import make_exclude_list

model_config = ConfigDict(arbitrary_types_allowed=True)

total_correct = 0
total_states = 0


@validate_call
def load_data(sessions: List[int], runs: List[int]):
	raws = []

	for session in sessions:
		raws += [
		read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True) for run in runs
	]

	raw = concatenate_raws(raws)

	raws = None

	standardize(raw)  # set channel names
	montage = make_standard_montage("standard_1005")
	# montage = read_custom_montage("./data/custom_fixture.txt")

	# excluded_channels = make_exclude_list(montage.ch_names)
	# raw.drop_channels(excluded_channels)

	raw.set_montage(montage)

	raw.annotations.rename(dict(T0="rest", T1="hands", T2="feet"))  # as documented on PhysioNet

	raw.notch_filter(60, method="iir")
	raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		proj=True,
		# event_id=["hands", "feet"],
		picks=picks,
		baseline=None,
		preload=True,
	)

	return epochs


@validate_call(config=model_config)
def split_datasets(epochs: Epochs, split_pct=0.8) -> Tuple[Epochs, Epochs]:
	# cutoff = int(len(epochs) * (split_pct / 100))

	# train = epochs[:-cutoff]
	# validate = epochs[-cutoff:]

	return (epochs, None)

def train_csp(epochs: Epochs) -> CSP:
	labels = epochs.events[:, -1] - 2
	data = epochs.get_data(copy=False)
	
	csp = CSP()

	print(len(labels), labels)
	print(len(data))

	csp.fit(data, labels)

	return csp


def fit(model, epochs: Epochs):
	global total_correct, total_states

	labels = epochs.events[:, -1] - 2
	data = epochs.get_data(copy=False)

	# csp = train_csp(epochs)
	csp = CSP(n_components=12)

	transformed = csp.fit_transform(data, labels)

	# print(transformed, transformed.shape, data.shape)

	# print("----")

	# print(labels, labels.shape)

	# print("----")

	# print("Transformed type:", type(transformed))
	# print("Transformed shape:", transformed.shape)
	# print("Labels shape:", labels.shape)
	# print("Data shape:", data.shape)

	model.partial_fit(transformed, labels, classes=[-1, 0, 1])


	results = model.predict(transformed)

	print(f"--- RESULTS ---")
	print(f"{100 * accuracy_score(labels, results):.2f}% correct.")
	print(f"{accuracy_score(labels, results, normalize=False):.2f} out of {len(labels)} correct.")
	print(f"--- RESULTS ---")

	total_correct += accuracy_score(labels, results, normalize=False)
	total_states += len(labels)


	return model

def train():
	global total_correct, total_states
	set_log_level(False)

	# train_datasets, test_dataset = split_datasets(epochs)

	csp = CSP(n_components=4)

	model = SGDClassifier(loss="modified_huber")

	# pipe = Pipeline([("CSP", csp), ("SGD", sgd)])

	for session_id in range(1, 110):
		print(session_id)
		epochs = load_data([session_id], [i for i in range(3, 15)]) #6, 10, 14

		model = fit(model, epochs)

	print("__--__--DONE--__--__")

	print(f"--- RESULTS ---")
	print(f"{100 * total_correct / total_states:.2f}% correct.")
	print(f"{total_correct:.2f} out of {total_states} correct.")
	print(f"--- RESULTS ---")



