
from mne import set_log_level
from mne.decoding import Scaler
import numpy as np
from pydantic import ConfigDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import pickle

from csp import CSP42
from utils import load_data

model_config = ConfigDict(arbitrary_types_allowed=True)

def train():
	set_log_level(False)

	total_accuracy = 0
	total_subjects=80

	superX = []
	superY = []

	for subject in range(1, total_subjects):
		print(f"=== SUBJECT #{subject:03d}")

		epochs = load_data([subject], [i for i in range(3, 15)])

		scaler = Scaler(epochs.info)
		csp = CSP42(n_components=10)
		model = LinearDiscriminantAnalysis(solver='lsqr')

		pipe = Pipeline([
			("Scaler", scaler),
			("CSP", csp),
			("EST", model)
		])

		# ---

		X = epochs.get_data(copy=False)
		Y = epochs.events[:, -1] - 1

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

		pipe.fit(X_train, Y_train)

		acc = pipe.score(X_test, Y_test)
		acc_train = pipe.score(X_train, Y_train)

		scores = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5, shuffle = True))
		print(f"Train Acc: {100 * acc_train:0.2f}% Test Accuracy: {acc * 100:0.2f}%, cross_val_score: {100 * scores.mean():0.2f}% ~ {100 * scores.std():0.2f}%")

		total_accuracy += acc

		with open(f"./models/S{subject:03d}.pkl", 'wb') as file:
			pickle.dump(pipe, file)

		print(f"=== SUBJECT #{subject:03d}\n")

		print(f"Total: {total_accuracy / (subject) * 100:0.2f}%")
