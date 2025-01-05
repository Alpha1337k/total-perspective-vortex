import glob
from matplotlib import pyplot as plt
import mne
from pydantic import validate_call

@validate_call
def plot_signals(outfile_base: str, raw):
	plt.clf()

	fig = plt.figure(figsize=(64, 1))

	for i in range(64):
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
	builtin_montages = mne.channels.get_builtin_montages(descriptions=True)

	# montage = mne.channels.make_standard_montage("biosemi64")
	montage = mne.channels.read_custom_montage("./data/custom_fixture.txt")

	montage.plot()

	plt.savefig(f"./results/headcap.png")
	plt.clf()

	for file in sorted(glob.glob("./data/S001/*.edf")):
		raw = mne.io.read_raw_edf(file, preload=True)

		raw.rename_channels(lambda n: n.replace('.', '').upper().replace('Z', 'z').replace('FP', 'Fp'))

		raw = raw.filter(l_freq=2.0, h_freq=75.0)

		print(raw)
		print(raw.info)

		# We are missing two channels. T10 T9
		raw.set_montage(montage, on_missing='ignore')

		raw.compute_psd().plot(picks="data", exclude="bads", amplitude=True)

		outfile_base = file.replace('data', 'results').replace('.edf', '')
		
		plt.savefig(f"{outfile_base}_psd.png")

		plot_signals(outfile_base, raw)

		print("\n\nPreProcessing")

		ica = mne.preprocessing.ICA(max_iter="auto", n_components=15)

		ica.fit(raw)

		ica.plot_properties(raw, picks=[0, 1])

		plt.savefig(f"{outfile_base}_ica.png")

		plt.clf()

		explained_var_ratio = ica.get_explained_variance_ratio(raw)
		for channel_type, ratio in explained_var_ratio.items():
			print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

		explained_var_ratio = ica.get_explained_variance_ratio(
			raw, components=[0], ch_type="eeg"
		)

		ica.plot_sources(raw, show_scrollbars=False)

		plt.savefig(f"{outfile_base}_ica_sources.png")

		plt.clf()

		ica.plot_components()

		plt.savefig(f"{outfile_base}_ica_components.png")

		plt.clf()

		ica.exclude = [0, 1]		

		ica.apply(raw)

		plot_signals(f"{outfile_base}_applied", raw)

