import glob
import mne
from typing import List
from matplotlib import pyplot as plt
from mne.io import read_raw_edf
from pydantic import validate_call
from mne.channels import get_builtin_montages, read_custom_montage, make_standard_montage

@validate_call
def is_target_neuron(id: str) -> bool:
	return True
	# return id.find('C') != -1 or id.find('F') != -1

@validate_call
def make_exclude_list(ch_names: List[str]) -> List[str]:
	return list(filter(lambda x: is_target_neuron(x) == False, ch_names))

@validate_call
def plot_signals(outfile_base: str, raw):
	plt.clf()

	fig = plt.figure(figsize=(64, 1))

	for i in range(len(raw.ch_names)):
		selection = raw[i, 0 : int(raw.info['sfreq']) * 60]

		x = selection[1]
		y = selection[0].T

		plt.subplot(64, 1, i + 1, title=f'{i} | {raw.ch_names[i]}')

		plt.plot(x, y)

	fig.set_figheight(25)
	fig.set_figwidth(25)

	fig.savefig(f"{outfile_base}_raw.png", bbox_inches='tight')

	plt.clf()

@validate_call
def plot(path: str):
	builtin_montages = get_builtin_montages(descriptions=True)

	montage = make_standard_montage("standard_1005")
	montage = read_custom_montage("./data/custom_fixture.txt")

	montage.plot()

	plt.savefig(f"./results/headcap.png")
	plt.clf()

	files = sorted(glob.glob("./data/S001/*.edf"))

	# files = ['./data/S001/S001R04.edf']

	excludes = make_exclude_list(montage.ch_names)

	for file in files:
		raw = read_raw_edf(file, preload=True)

		raw.rename_channels(lambda n: n.replace('.', '').upper().replace('Z', 'z').replace('FP', 'Fp'))

		raw = raw.filter(l_freq=2.0, h_freq=75.0, fir_design="firwin", skip_by_annotation="edge")

		print(raw)
		print(raw.info)

		raw.notch_filter(60)

		raw.drop_channels(excludes)

		raw.set_montage(montage)

		raw.compute_psd().plot(picks="data", amplitude=True)

		outfile_base = file.replace('data', 'results').replace('.edf', '')
		
		plt.savefig(f"{outfile_base}_psd.png")

		plot_signals(outfile_base, raw)

		print("\n\nPreProcessing")

		ica = mne.preprocessing.ICA(max_iter="auto", n_components=10)

		ica.fit(raw)

		ica.plot_properties(raw, picks=[0, 1])

		plt.savefig(f"{outfile_base}_ica.png")

		plt.clf()

		explained_var_ratio = ica.get_explained_variance_ratio(raw)
		for channel_type, ratio in explained_var_ratio.items():
			print(f"Fraction of {channel_type} variance explained by all components: {ratio}")


		ica.plot_sources(raw, show_scrollbars=False, stop=60)

		plt.savefig(f"{outfile_base}_ica_sources.png")

		plt.clf()

		ica.plot_overlay(raw, exclude=[0, 1], picks="eeg")

		plt.savefig(f"{outfile_base}_ica_exclude.png")

		plt.clf()

		ica.plot_components()

		plt.savefig(f"{outfile_base}_ica_components.png")

		plt.clf()

		ica.apply(raw, exclude = [0, 1])

		plot_signals(f"{outfile_base}_applied", raw)

		plt.savefig(f"{outfile_base}_ica_components.png")

		plt.clf()

