import glob
from matplotlib import pyplot as plt
import mne
from pydantic import validate_call

@validate_call
def plot_signals(outfile_base: str, raw):
	plt.clf()

	fig = plt.figure(figsize=(64, 1))

	for i in range(64):
		selection = raw[i]

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

	easycap_montage = mne.channels.make_standard_montage("easycap-M1")

	easycap_montage.plot()

	print(sorted(easycap_montage.ch_names))

	plt.savefig(f"./results/headcap.png")
	plt.clf()

	for file in sorted(glob.glob("./data/S001/*.edf")):
		raw = mne.io.read_raw_edf(file, preload=True)

		print(sorted(raw.ch_names))

		raw.rename_channels(lambda n: n.replace('.', '').upper().replace('Z', 'z'))

		raw.filter(l_freq=1.0, h_freq=None)

		print(raw)
		print(raw.info)

		# raw.set_montage(easycap_montage)

		raw.compute_psd().plot(picks="data", exclude="bads", amplitude=True)

		outfile_base = file.replace('data', 'results').replace('.edf', '')
		
		plt.savefig(f"{outfile_base}_psd.png")

		plot_signals(outfile_base, raw)

		print("\n\nPreProcessing")

		ica = mne.preprocessing.ICA(max_iter=400)

		ica.fit(raw)

		ica.plot_properties(raw, picks=[1,2])

		plt.savefig(f"{outfile_base}_ica.png")

		plt.clf()

		exit(1)

