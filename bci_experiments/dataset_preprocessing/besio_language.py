import bci_experiments
import bci_experiments.constants as consts

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

besio_dataset_location = consts.DATSET_BESIO_LANGUAGE_PATH
file_name = 'ID01/7-3-19/Raw_Files/LM_DV7319_2.vhdr'
raw = mne.io.read_raw_brainvision(os.path.join(consts.DATASET_LOCATION, besio_dataset_location, file_name), preload=True)

import pdb; pdb.set_trace()
eeg = raw.get_data().T

plt.figure(1)
n = 5000
plt.clf()
plt.subplot(2, 1, 1)
plt.plot(eeg[:n,::2] + np.arange(20) * 0.0005)

plt.subplot(2, 1, 2)
plt.plot(eeg[:n,1::2] + np.arange(20) * 0.00005)

plt.show(block=True)
