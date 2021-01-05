# coding=utf-8
import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from bcikit.datasets import EEGDataset


class OpenBMISSVEP(EEGDataset):
    """
    This is a SSVEP dataset a four target frequencies.
    EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy.
    (Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee, John Williamson, Siamac Fazli, Seong-Whan Lee.)
    https://academic.oup.com/gigascience/article/8/5/giz002/5304369
    
    Args:
        root (string): Root directory of the dataset.
        subject_id (int): Subject ID
        session (int): Session number
        verbose (bool, optional): If ``True``, Print logs for debugging. Default: ``False``
    
    Usage:
        from bcikit.datasets.ssvep import OpenBMISSVEP
        dataset = OpenBMISSVEP(root="_data/openbmissvep", subject_id=1, session=1, verbose=True)
        print(dataset.data.shape)
        print(dataset.targets)

    Target frequencies: [12.0,8.57,6.67,5.45]
    Sampling rate: 1000 Hz
    """

    def __init__(self, root: str, subject_id: int, session: int, verbose: bool = False) -> None:

        self.sample_rate = 1000
        self.data, self.targets, self.channel_names = self._load_data(
            root, subject_id, session, verbose)

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self, root, subject_id, session, verbose):

        path = os.path.join(root, 'session'+str(session),
                            's'+str(subject_id)+'/EEG_SSVEP.mat')

        data_mat = sio.loadmat(path)

        objects_in_mat = []
        for i in data_mat['EEG_SSVEP_train'][0][0]:
            objects_in_mat.append(i)

        # data
        data = objects_in_mat[0][:, :, :].copy()
        data = np.transpose(data, (1, 2, 0))
        data = data.astype(np.float32)

        # label
        targets = []
        for i in range(data.shape[0]):
            targets.append([objects_in_mat[2][0][i], 0, objects_in_mat[4][0][i]])
        targets = np.array(targets)
        targets = targets[:, 2]
        targets = targets-1

        # channel
        channel_names = [v[0] for v in objects_in_mat[8][0]]

        data = np.expand_dims(data, axis=0) # expand dims from (trial,channel,time) to (session,trial,channel,time)

        if verbose:
            print('Load path:', path)
            print('Objects in .mat', len(objects_in_mat),
                data_mat['EEG_SSVEP_train'].dtype.descr)
            print()
            print('Data shape', data.shape)
            print('Targets shape', targets.shape)

        return data, targets, channel_names
