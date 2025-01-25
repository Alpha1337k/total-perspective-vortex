
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
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import pickle

from csp import CSP42
from plot import make_exclude_list
from utils import load_data

model_config = ConfigDict(arbitrary_types_allowed=True)

def train():
	set_log_level(False)

	for subject in range(1, 110):
		print(f"=== SUBJECT #{subject:03d}")

		epochs = load_data([subject], [i for i in range(3, 15)])

		scaler = Scaler(epochs.info)
		csp = CSP42(n_components=8)
		model = LinearDiscriminantAnalysis(solver='lsqr')

		pipe = Pipeline([("Scaler", scaler), ("CSP", csp), ("EST", model)])

		# ---

		X = epochs.get_data(copy=False)
		Y = epochs.events[:, -1] - 1

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		pipe.fit(X_train, Y_train)

		acc = pipe.score(X_test, Y_test)
		scores = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5, shuffle = True))
		print(f"Test Accuracy: {acc * 100:0.2f}%, cross_val_score: {100 * scores.mean():0.2f}% ~ {100 * scores.std():0.2f}%")

		with open(f"./models/S{subject:03d}.pkl", 'wb') as file:
			pickle.dump(pipe, file)

		print(f"=== SUBJECT #{subject:03d}\n")






