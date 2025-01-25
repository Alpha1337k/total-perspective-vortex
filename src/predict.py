

import pickle

from mne import set_log_level
import numpy as np
from pydantic import validate_call
from sklearn.pipeline import Pipeline

from utils import load_data


@validate_call
def predict(subject: int, recording: int, super: bool):
	pipe: Pipeline
	
	if super:
		pipe = pickle.load(open(f"./models/Super.pkl", 'rb'))
	else:
		pipe = pickle.load(open(f"./models/{subject:02d}.pkl", 'rb'))

	set_log_level(False)

	epochs = load_data([subject], [recording])

	X = epochs.get_data(copy=False).astype(np.float64)
	Y = epochs.events[:, -1] - 1

	n_correct = 0

	predictions = pipe.predict(X)

	for i, (prediction, y) in enumerate(zip(predictions, Y)):


		print(f"epoch {i:02d}:\tReal: [{y}]\t Pred: {prediction}\t{prediction == y}")

		if prediction == y:
			n_correct += 1

	print(f"Accuracy: {n_correct / len(Y) * 100:0.2f}%")
