
from typing import Any, List, Tuple
from mne import Epochs, concatenate_raws, pick_types
from mne.channels import make_standard_montage
from mne.decoding import CSP
from mne.io import BaseRaw, read_raw_edf
import numpy as np
from pydantic import ConfigDict, validate_call
from mne.datasets.eegbci import standardize
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline

model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call
def load_data(sessions: List[int], runs: List[int]):
	raw: BaseRaw | Any = None

	for session in sessions:
		raw = concatenate_raws([
			read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True) for run in runs
		])

	standardize(raw)  # set channel names
	montage = make_standard_montage("standard_1005")
	raw.set_montage(montage)
	raw.annotations.rename(dict(T0="rest", T1="hands", T2="feet"))  # as documented on PhysioNet

	raw.notch_filter(60, method="iir")
	raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		proj=True,
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


def fit(pipe: Pipeline, epochs: Epochs) -> Pipeline:
	labels = epochs.events[:, -1] - 2
	data = epochs.get_data(copy=False)

	# csp = train_csp(epochs)
	csp = CSP()

	model = LogisticRegression()

	transformed = csp.fit_transform(data, labels)

	print(transformed, transformed.shape, data.shape)

	print("----")

	print(labels, labels.shape)

	print("----")

	print(transformed[0])

	print("Transformed type:", type(transformed))
	print("Transformed shape:", transformed.shape)
	print("Labels shape:", labels.shape)

	model.fit(transformed, labels)


	results = model.predict(transformed)

	print(results)


	return model

def train():
	epochs = load_data([1], [6]) #6, 10, 14

	train_datasets, test_dataset = split_datasets(epochs)

	csp = CSP(n_components=4)

	sgd = SGDRegressor()

	model = Pipeline([("CSP", csp), ("SGD", sgd)])

	fit(model, train_datasets)



