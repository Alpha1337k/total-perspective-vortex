

from io import BufferedReader
import pickle

from mne import set_log_level
from sklearn.pipeline import Pipeline

from utils import load_data

experiments = [
	[4, 6, 8, 10, 12, 14],
	[3, 5, 7, 9, 11, 13],
	[6, 10, 14],
	[5, 9, 13],
	[4, 8, 12],
	[3, 7, 11],
]


def score():
	set_log_level(False)

	accuracies = [0. for i in range(len(experiments))]


	for experiment_id, experiment in enumerate(experiments):
		for subject in range(1, total_subjects):
			pipe: Pipeline = pickle.load(open(f'./models/S{subject:03d}.pkl', 'rb'))

			epochs = load_data([subject], [i for i in range(3, 15)])
			
			X = epochs.get_data(copy=False)
			Y = epochs.events[:, -1] - 1

			acc = pipe.score(X, Y)

			print(f"experiment {experiment_id}: subject {subject:03d}: accuracy = {acc * 100:0.2f}%")

			accuracies[experiment_id] += float(acc)

		print(f"experiment {experiment_id}: mean accuracy = {accuracies[experiment_id] / total_subjects * 100:0.2f}%" )

	print("\n-=-=-=-=-=-")

	for experiment_id in range(len(experiments)):
		print(f"experiment {experiment_id}: mean accuracy = {accuracies[experiment_id] / total_subjects * 100:0.2f}%" )

	total = sum(accuracies) / (total_subjects * 6) * 100

	print(f"\nMean accuracy of 6 experiments = {total:0.2f}%")

	print("-=-=-=-=-=-")