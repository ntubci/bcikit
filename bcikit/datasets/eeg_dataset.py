# coding=utf-8
from torch.utils.data import Dataset
import numpy as np
from typing import Any, Callable, List, Optional, Tuple


class EEGDataset(Dataset):
    """
    This is an abstract class for all EEG dataset in this folder.

    `self.data`:
        should be Numpy's ndarray: (trial, channel, time)

    Extend:
        Extend class to `EEGDataset`:
        ```
        from bcikit.datasets import EEGDataset
        class DatasetName(EEGDataset):
        ```

    Init function: 
        Required params are `root`, `subject_id`, `verbose`, and `**kwargs`. 
        If this dataset split sessions in multiple .mat file, add `session` param. Add any additional params accordingly.
        ```
        def __init__(self, root: str, subject_id: int, session: int, verbose: bool = False, **kwargs) -> None:
        ```
    
    """
    def __init__(self, 
            data: np.ndarray, 
            targets: np.ndarray, 
            channel_names: [] = None, 
            sample_rate: int = None
        ):
        self.data = data.astype(np.float32)
        self.targets = targets
        self.sample_rate = sample_rate
        self.channel_names = channel_names

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self, root: str) -> None:
        """
        Implement this function in the dataset class to extract data from files.
        """
        raise NotImplementedError
                
    def recommended_preprocessing(self, 
                                  data: np.ndarray, 
                                  targets: np.ndarray, 
                                  channel_names: [] = None, 
                                  sample_rate: int = None,
                                  verbose: bool = False, 
                                  **kwargs: Any
                                 ):
        """
        Implement this function in the dataset class for standard preprocessing described by the dataset paper.
        `do_recommended_preprocessing=True` in `EEGDataloader` will perform preprocessing.
        Can use this function as template.
        """
        print("recommended_preprocessing not implemented")
        return data, targets

    def set_channel_names(self, channel_names) -> List[str]:
        self.channel_names = channel_names
