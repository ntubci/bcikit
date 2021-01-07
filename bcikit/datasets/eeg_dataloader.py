# coding=utf-8
from torch.utils.data import DataLoader
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

from bcikit.datasets import EEGDataset
from bcikit.dataclass import BcikitConfig


class EEGDataloader(EEGDataset):
    """
    Load multiple .mat files into a single EEGDataset 
    where `data` is numpy.ndarray, (subject,session,trial,channel,time) 
    and `targets` is (subject,session,trial).
    So we can load all the data, and perform all the preprocessing once.

    Args:
        dataset (EEGDataset): EEGDataset class to load the data
        config (BcikitConfig): A standardize config
        preprocessing_fn (function, optional): Default: ``None``.
            A function to perform a preprocessing on `data` and `targets`, then return `data` and `targets`.
            ``None`` if do not have customized preprocessing steps.
            Example, customize your preprocessing function:
                ```
                def preprocessing_fn(data, targets, **kwargs)
                    print("Data shape", data.shape) # (subject,session,trial,channel,time)
                    # ... does preprocessing here
                    return data, targets
                ```
        data_selection_fn (function, optional): Default: ``None``.
            A function that does data selection and return PyTorch dataloaders.
            You can customize your data selection function or use common ones from `data_selection_methods`.
            TODO: ``None`` for default data selection
            Example, customize your data selection function:
                ```
                def data_selection_fn(batchsize=32, **kwargs):
                    # ... select and slice data here
                    return train_loader, val_loader, test_loader # these are PyTorch DataLoader objects
                ```
    """
    def __init__(self, 
                 dataset: EEGDataset, 
                 config: BcikitConfig,
                 preprocessing_fn = None,
                 data_selection_fn = None,
                 **kwargs: Any
    ) -> None:

        self.subject_ids = config.dataset.subject_ids # these could be referenced in `get_dataloaders`
        self.sessions = config.dataset.sessions # these could be referenced in `get_dataloaders`
        self.data_selection_fn = data_selection_fn
        
        # `data` is numpy.ndarray, (subject,session,trial,channel,time)
        # `targets` is (subject,session,trial)
        data, targets, channel_names, sample_rate = self._load_data(
            root=config.dataset.root, 
            dataset=dataset, 
            subject_ids=config.dataset.subject_ids, 
            sessions=config.dataset.sessions,
            verbose=config.log_level,
            **kwargs
        )
        
        if config.dataset.do_recommended_preprocessing:
            data, targets = dataset.recommended_preprocessing(
                self,
                data=data, 
                targets=targets, 
                channel_names=channel_names, 
                sample_rate=sample_rate, 
                verbose=config.log_level, 
                **kwargs
            )

        if preprocessing_fn is not None:
            data, targets = preprocessing_fn(
                data=data, 
                targets=targets, 
                channel_names=channel_names, 
                sample_rate=sample_rate, 
                verbose=config.log_level, 
                **kwargs
            )

        super().__init__(data, targets, channel_names, sample_rate)
    
    def get_dataloaders(self, **kwargs: Any):

        if self.data_selection_fn is not None:
            return self.data_selection_fn(
                self=self,
                **kwargs
            )
        
        else:
            # TODO: implement a default
            raise NotImplementedError
    
    
    def _load_data(self, root, dataset: EEGDataset, subject_ids: [], sessions: [], verbose: bool = False, **kwargs) -> None:
        num_subjects = len(subject_ids)
        num_sessions = len(sessions) if sessions is not None else 1
        sessions = sessions if sessions is not None else [None]

        subjects_data = None
        subjects_target = None
        channel_names = None
        sample_rate = None

        for sub_idx, subject_id in enumerate(subject_ids):
            if verbose>=1:
                print('Load subject:', subject_id)
            
            for ses_idx, session in enumerate(sessions):

                # data
                dataset_single = dataset(root=root, subject_id=subject_id, session=session, **kwargs)
                single_data = np.expand_dims(dataset_single.data, axis=(0, 1))

                if subjects_data is None:
                    subjects_data = np.zeros((num_subjects, num_sessions, single_data.shape[2], single_data.shape[3], single_data.shape[4]))
                    subjects_target = np.zeros((num_subjects, num_sessions, single_data.shape[2]))

                    channel_names = dataset_single.channel_names
                    sample_rate = dataset_single.sample_rate

                subjects_data[sub_idx,ses_idx,:,:,:] = single_data

                # targets
                single_targets = np.expand_dims(dataset_single.targets, axis=(0, 1))
                subjects_target[sub_idx,ses_idx,:] = single_targets
        
        return subjects_data, subjects_target, channel_names, sample_rate

    
    
    
# # coding=utf-8
# from torch.utils.data import DataLoader
# from typing import Any, Callable, List, Optional, Tuple
# import numpy as np

# from bcikit.datasets import EEGDataset


# class EEGDataloader(EEGDataset):
#     """
#     Load multiple .mat files into a single EEGDataset 
#     where `data` is numpy.ndarray, (subject,session,trial,channel,time) 
#     and `targets` is (subject,session,trial).
#     So we can load all the data, and perform all the preprocessing once.

#     Args:
#         dataset (EEGDataset): EEGDataset class to load the data
#         root (string): Root directory of the dataset.
#         subject_id (int): Subject ID
#         sessions (list, optional): A list of session ID. Default: ``None``
#         do_recommended_preprocessing (bool, optional): If ``True``, do preprocessing according to the dataset paper if `recommended_preprocessing` function is implemented in `dataset`
#         preprocessing_fn (function, optional): Default: ``None``.
#             A function to perform a preprocessing on `data` and `targets`, then return `data` and `targets`.
#             ``None`` if do not have customized preprocessing steps.
#             Example, customize your preprocessing function:
#                 ```
#                 def preprocessing_fn(data, targets, **kwargs)
#                     print("Data shape", data.shape) # (subject,session,trial,channel,time)
#                     # ... does preprocessing here
#                     return data, targets
#                 ```
#         data_selection_fn (function, optional): Default: ``None``.
#             A function that does data selection and return PyTorch dataloaders.
#             You can customize your data selection function or use common ones from `data_selection_methods`.
#             TODO: ``None`` for default data selection
#             Example, customize your data selection function:
#                 ```
#                 def data_selection_fn(batchsize=32, **kwargs):
#                     # ... select and slice data here
#                     return train_loader, val_loader, test_loader # these are PyTorch DataLoader objects
#                 ```
#         verbose (bool, optional): If ``True``, print logs for debugging. Default: ``False``

    
#     - Assuming that every .mat files contain the number of trials, channels and time. Is that true? Would there any dataset, where every subject has different number of trials?
#     """
#     def __init__(self, 
#         dataset: EEGDataset, 
#         root: str, 
#         subject_ids: [], 
#         sessions: [] = None,
#         do_recommended_preprocessing: bool = False, 
#         preprocessing_fn = None,
#         data_selection_fn = None,
#         verbose: bool = False, 
#         **kwargs: Any
#     ) -> None:

#         self.subject_ids = subject_ids
#         self.sessions = sessions
#         self.data_selection_fn = data_selection_fn
#         self.verbose = verbose
        
#         # `data` is numpy.ndarray, (subject,session,trial,channel,time)
#         # `targets` is (subject,session,trial)
#         data, targets, channel_names, sample_rate = self._load_data(
#             root=root, 
#             dataset=dataset, 
#             subject_ids=subject_ids, 
#             sessions=sessions,
#             verbose=verbose,
#             **kwargs
#         )
        
#         if do_recommended_preprocessing:
#             data, targets = dataset.recommended_preprocessing(
#                 self,
#                 data=data, 
#                 targets=targets, 
#                 channel_names=channel_names, 
#                 sample_rate=sample_rate, 
#                 verbose=verbose, 
#                 **kwargs
#             )

#         if preprocessing_fn is not None:
#             data, targets = preprocessing_fn(
#                 data=data, 
#                 targets=targets, 
#                 channel_names=channel_names, 
#                 sample_rate=sample_rate, 
#                 verbose=verbose, 
#                 **kwargs
#             )

#         super().__init__(data, targets, channel_names, sample_rate)
    
#     def get_dataloaders(self, **kwargs: Any):

#         if self.data_selection_fn is not None:
#             return self.data_selection_fn(
#                 self=self,
#                 **kwargs
#             )
        
#         else:
#             # TODO: implement a default
#             raise NotImplementedError
    
    
#     def _load_data(self, root, dataset: EEGDataset, subject_ids: [], sessions: [], verbose: bool = False, **kwargs) -> None:
#         num_subjects = len(subject_ids)
#         num_sessions = len(sessions) if sessions is not None else 1
#         sessions = sessions if sessions is not None else [None]

#         subjects_data = None
#         subjects_target = None
#         channel_names = None
#         sample_rate = None

#         for sub_idx, subject_id in enumerate(subject_ids):
#             if verbose:
#                 print('Load subject:', subject_id)
            
#             for ses_idx, session in enumerate(sessions):

#                 # data
#                 dataset_single = dataset(root=root, subject_id=subject_id, session=session, **kwargs)
#                 single_data = np.expand_dims(dataset_single.data, axis=(0, 1))

#                 if subjects_data is None:
#                     subjects_data = np.zeros((num_subjects, num_sessions, single_data.shape[2], single_data.shape[3], single_data.shape[4]))
#                     subjects_target = np.zeros((num_subjects, num_sessions, single_data.shape[2]))

#                     channel_names = dataset_single.channel_names
#                     sample_rate = dataset_single.sample_rate

#                 subjects_data[sub_idx,ses_idx,:,:,:] = single_data

#                 # targets
#                 single_targets = np.expand_dims(dataset_single.targets, axis=(0, 1))
#                 subjects_target[sub_idx,ses_idx,:] = single_targets
        
#         return subjects_data, subjects_target, channel_names, sample_rate
