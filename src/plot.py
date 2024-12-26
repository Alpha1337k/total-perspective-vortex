import glob
from matplotlib import pyplot as plt
import mne
from pydantic import validate_call


@validate_call
def plot(path: str):
	builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
	for montage_name, montage_description in builtin_montages:
		print(f"{montage_name}: {montage_description}")

	easycap_montage = mne.channels.make_standard_montage("easycap-M1")
	print(easycap_montage)

	easycap_montage.plot()
	plt.savefig(f"./results/headcap.png")
	plt.clf()

	for file in sorted(glob.glob("./data/S001/*.edf")):
		raw = mne.io.read_raw_edf(file)

		raw.rename_channels(lambda n: n.replace('.', '').upper().replace('Z', 'z'))


		print(raw)
		print(raw.info)

		raw.set_montage(easycap_montage, on_missing="ignore")

		raw.compute_psd().plot(picks="data", exclude="bads", amplitude=True)

		outfile_base = file.replace('data', 'results').replace('.edf', '')

		
		plt.savefig(f"{outfile_base}_psd.png")

		plt.clf()

		fig = plt.figure(figsize=(64, 1))

		for i in range(64):
			selection = raw[i, 0: 60 * raw.info['sfreq']]

			x = selection[1]
			y = selection[0].T

			plt.subplot(64, 1, i + 1, title=f'{i} | {raw.ch_names[i]}')

			plt.plot(x, y)

		fig.set_figheight(25)
		fig.set_figwidth(25)

		fig.savefig(f"{outfile_base}_raw.png", bbox_inches='tight')

		plt.clf()

