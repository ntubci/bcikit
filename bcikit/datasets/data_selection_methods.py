# coding=utf-8
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

from bcikit.datasets import EEGDataset


def leave_one_subject_out(self, test_subject_id=1, batchsize=32, kfold_k=0, kfold_split=3, **kwargs):
    """
    Since this is focus on subject, data should be preprocessed to (subject, trial, channel, time).
    This function select `test_subject_id` for test_loader, and creates `train_loader` and `val_loader` with stratified k-fold.
    """

    sub_idx = self.subject_ids.index(test_subject_id)

    # select test
    selected_subject_data = self.data[sub_idx, :, :, :]
    selected_subject_targets = self.targets[sub_idx, :]
    test_dataset = EEGDataset(selected_subject_data, selected_subject_targets)

    # select train and val
    indices = np.arange(self.data.shape[0])
    other_subjects_data = self.data[indices!=sub_idx, :, :, :]
    other_subjects_targets = self.targets[indices!=sub_idx, :]

    other_subjects_data = other_subjects_data.reshape((other_subjects_data.shape[0]*other_subjects_data.shape[1], other_subjects_data.shape[2], other_subjects_data.shape[3]))
    other_subjects_targets = other_subjects_targets.reshape((other_subjects_targets.shape[0]*other_subjects_targets.shape[1]))

    # k-fold cross validation
    (X_train, y_train), (X_val, y_val) = dataset_split_stratified(other_subjects_data, other_subjects_targets, k=kfold_k, n_splits=kfold_split, seed=71, shuffle=True, EEGDataset_class=None)
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
    return train_loader, val_loader, test_loader


def dataset_split_stratified(X, y, k=0, n_splits=3, seed=71, shuffle=True, EEGDataset_class=None):
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=shuffle)
    split_data = skf.split(X, y)

    for idx, value in enumerate(split_data):

        if k != idx:
            continue
        else:
            train_index, test_index = value
        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            return (X_train, y_train), (X_test, y_test)
