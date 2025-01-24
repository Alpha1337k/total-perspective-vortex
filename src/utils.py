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

	for session in sessions:
		# print(f"LOADING session #{session} ({len(raws)})")

		new_raws = [
			read_raw_edf(f"./data/S{session:03d}/S{session:03d}R{run:02d}.edf", preload=True) for run in runs
		]


		# print([r.info['sfreq'] for r in new_raws])

		raws += new_raws

	raw = concatenate_raws(raws)

	raws = None

	standardize(raw)  # set channel names
	# montage = make_standard_montage("standard_1005")
	montage = read_custom_montage("./data/custom_fixture.txt")

	excluded_channels = make_exclude_list(montage.ch_names)

	# print(excluded_channels)

	raw.drop_channels(excluded_channels)

	raw.set_montage(montage)

	raw.annotations.rename(dict(T0="rest", T1="hands", T2="feet"))  # as documented on PhysioNet

	raw.set_eeg_reference(projection=True)

	raw.notch_filter(60, method="iir")
	raw.filter(7.0, 32.0, 
			fir_design="firwin", 
			)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		proj=True,
		event_id=["hands", "feet"],
		picks=picks,
		baseline=None,
		# detrend=1,
		# tmin=1,
		# tmax=2,
		preload=True,
	)

	return epochs