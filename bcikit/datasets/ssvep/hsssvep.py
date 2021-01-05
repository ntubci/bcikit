# coding=utf-8
import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from bcikit.datasets import EEGDataset


class HSSSVEP(EEGDataset):
    """
    A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces
    Yijun Wang, Xiaogang Chen, Xiaorong Gao, Shangkai Gao
    https://ieeexplore.ieee.org/document/7740878

    Args:
        root (string): Root directory of the dataset.
        subject_id (int): Subject ID
        verbose (bool, optional): If ``True``, Print logs for debugging. Default: ``False``
    
    Usage:
        from bcikit.datasets.ssvep import HSSSVEP
        dataset = HSSSVEP(root="_data/hsssvep", subject_id=1, verbose=True)
        print(dataset.data.shape)

    Sampling rate: 250 Hz
    Targets: [8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8]
    Channel: ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

    This dataset gathered SSVEP-BCI recordings of 35 healthy subjects (17 females, aged 17-34 years, mean age: 22 years) focusing on 40 characters flickering at different frequencies (8-15.8 Hz with an interval of 0.2 Hz). For each subject, the experiment consisted of 6 blocks. Each block contained 40 trials corresponding to all 40 characters indicated in a random order. Each trial started with a visual cue (a red square) indicating a target stimulus. The cue appeared for 0.5 s on the screen. Subjects were asked to shift their gaze to the target as soon as possible within the cue duration. Following the cue offset, all stimuli started to flicker on the screen concurrently and lasted 5 s. After stimulus offset, the screen was blank for 0.5 s before the next trial began, which allowed the subjects to have short breaks between consecutive trials. Each trial lasted a total of 6 s. To facilitate visual fixation, a red triangle appeared below the flickering target during the stimulation period. In each block, subjects were asked to avoid eye blinks during the stimulation period. To avoid visual fatigue, there was a rest for several minutes between two consecutive blocks.
    EEG data were acquired using a Synamps2 system (Neuroscan, Inc.) with a sampling rate of 1000 Hz. The amplifier frequency passband ranged from 0.15 Hz to 200 Hz. Sixty-four channels covered the whole scalp of the subject and were aligned according to the international 10-20 system. The ground was placed on midway between Fz and FPz. The reference was located on the vertex. Electrode impedances were kept below 10 K". To remove the common power-line noise, a notch filter at 50 Hz was applied in data recording. Event triggers generated by the computer to the amplifier and recorded on an event channel synchronized to the EEG data. 
    The continuous EEG data was segmented into 6 s epochs (500 ms pre-stimulus, 5.5 s post-stimulus onset). The epochs were subsequently downsampled to 250 Hz. Thus each trial consisted of 1500 time points. Finally, these data were stored as double-precision floating-point values in MATLAB and were named as subject indices (i.e., S01.mat, ", S35.mat). For each file, the data loaded in MATLAB generate a 4-D matrix named "data" with dimensions of [64, 1500, 40, 6]. The four dimensions indicate "Electrode index", "Time points", "Target index", and "Block index". The electrode positions were saved in a "64-channels.loc" file. Six trials were available for each SSVEP frequency. Frequency and phase values for the 40 target indices were saved in a "Freq_Phase.mat" file.
    Information for all subjects was listed in a "Sub_info.txt" file. For each subject, there are five factors including "Subject Index", "Gender", "Age", "Handedness", and "Group". Subjects were divided into an "experienced" group (eight subjects, S01-S08) and a "naive" group (27 subjects, S09-S35) according to their experience in SSVEP-based BCIs.
    """
    def __init__(self, root: str, subject_id: int, verbose: bool = False, **kwargs) -> None:

        sample_rate = 250
        data, targets, channel_names = self._load_data(root, subject_id, verbose)
        super().__init__(data, targets, channel_names, sample_rate)

    def _load_data(self, root, subject_id, verbose):

        path = os.path.join(root, 'S'+str(subject_id)+'.mat')
        data_mat = sio.loadmat(path)

        raw_data = data_mat['data'].copy()
        raw_data = np.transpose(raw_data, (2,3,0,1))

        data = []
        targets = []
        for target_id in np.arange(raw_data.shape[0]):
            data.extend(raw_data[target_id])
            this_target = np.array([target_id]*raw_data.shape[1])
            targets.extend(this_target)
        
        # Each trial started with a 0.5-s target cue. Subjects were asked to shift their gaze to the target as soon as possible. After the cue, all stimuli started to flicker on the screen concurrently for 5 s. Then, the screen was blank for 0.5 s before the next trial began. Each trial lasted 6 s in total.
        # We start from 160, because 0.5s Cue + 0.14s (visual latency) as they use phase in stimulus presentation. 0.64*250 = 160
        # We also cut the signal off after 4 seconds
        # data = np.array(data)[:,:,160:1160]
        data = np.array(data)[:,:,250:1250] # better data quality
        # data = np.expand_dims(data, axis=0) # expand dims from (trial,channel,time) to (session,trial,channel,time)

        targets = np.array(targets)

        channel_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

        if verbose:
            print('Load path:', path)
            print('Data shape', data.shape)
            print('Targets shape', targets.shape)

        return data, targets, channel_names
