from typing import List

from mne import Epochs, concatenate_raws, pick_types
from mne.channels import read_custom_montage
from mne.datasets.eegbci import standardize
from mne.io import read_raw_edf
from pydantic import ConfigDict, validate_call

from plot import make_exclude_list


model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call
def load_data(sessions: List[int], runs: List[int]):
	raws = []

	montage = read_custom_montage("./data/custom_fixture.txt")

	excluded_channels = make_exclude_list(montage.ch_names)

	# print("Loading sessions")

	for session in sessions:
		# print(f"LOADING session #{session} ({len(raws)})")

		for run in runs:
			raw = read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True)

			standardize(raw)
			raw.drop_channels(excluded_channels)
			raw.annotations.rename(dict(T0="rest", T1="hands", T2="feet"))  # as documented on PhysioNet

			raw.set_eeg_reference(projection=True)

			raw.notch_filter(60, method="iir")
			raw.filter(7.0, 32.0, 
					fir_design="firwin", 
					)
			raws.append(raw)

	# print("Done loading sessions. Concatenating + splitting.")

	raw = concatenate_raws(raws)

	raws = None

	# print(excluded_channels)
	raw.set_montage(montage)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		proj=True,
		event_id=["hands", "feet"],
		picks=picks,
		baseline=None,
		detrend=0,
		tmin=0.2,
		tmax=2.2,
		preload=True,
	)

	# print("Splitted!")

	return epochs