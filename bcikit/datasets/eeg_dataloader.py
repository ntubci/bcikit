# coding=utf-8
from torch.utils.data import DataLoader
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

from bcikit.datasets import EEGDataset


class EEGDataloader():
    """
    
    """
    def __init__(self, 
        dataset: EEGDataset, 
        root: str, 
        subject_ids: [], 
        sessions: [] = None,
        verbose: bool = False, 
    ) -> None:

        pass
