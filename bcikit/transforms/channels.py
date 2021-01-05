# coding=utf-8
import numpy as np


def pick_channels(data: np.ndarray,
                  channel_names: [str],
                  selected_channels: [str],
                  verbose: bool = False) -> np.ndarray:
    """
    Given a list of channels names from the dataset, and a list of channel names for selection, Filter channels and remove the others.

    Args:
        data (np.ndarray): 3 dim (trial,channel,time) or 4-dim (X, trial,channel,time) or 5 dim (subject,session,trial,channel,time) Numpy array.
        channel_names (list): A list of channel names in the dataset, like ['FP1', 'FPZ', 'FP2', 'AF3', ... , 'O2', 'CB2']
        selected_channels (list): A list of channel names for selection, like ['FP1', 'FPZ', 'FP2', 'AF3', ... , 'O2', 'CB2']
        verbose (bool, optional): If ``True``, Print logs for debugging. Default: ``False``
    """

    assert len(data.shape) == 3 or len(data.shape) == 5, "data shape must be (trial,channel,time) or (subject,session,trial,channel,time)"

    picked_ch = pick_channels_mne(channel_names, selected_channels)
    
    if len(data.shape) == 3: # (trial,channel,time)
        data = data[:,  picked_ch, :]
    elif len(data.shape) == 4: # (X, trial,channel,time)
        data = data[:,  picked_ch, :]
    elif len(data.shape) == 5: # (subject,session,trial,channel,time)
        data = data[:, :, :, picked_ch, :]

    if verbose:
        print('all channels: ', len(channel_names), channel_names)
        print('picked channel index', picked_ch)
        print()

    del picked_ch

    return data


def pick_channels_mne(ch_names, include, exclude=[], ordered=False):
    """Pick channels by names.
    Returns the indices of ``ch_names`` in ``include`` but not in ``exclude``.
    Taken from https://github.com/mne-tools/mne-python/blob/master/mne/io/pick.py

    Args:
        ch_names : list of str
            List of channels.
        include : list of str
            List of channels to include (if empty include all available).
            .. note:: This is to be treated as a set. The order of this list
            is not used or maintained in ``sel``.
        exclude : list of str
            List of channels to exclude (if empty do not exclude any channel).
            Defaults to [].
        ordered : bool
            If true (default False), treat ``include`` as an ordered list
            rather than a set, and any channels from ``include`` are missing
            in ``ch_names`` an error will be raised.
            .. versionadded:: 0.18
    
    Returns:
        sel : array of int
            Indices of good channels.

    See Also:
        pick_channels_regexp, pick_types
    """
    if len(np.unique(ch_names)) != len(ch_names):
        raise RuntimeError('ch_names is not a unique list, picking is unsafe')
    # _check_excludes_includes(include)
    # _check_excludes_includes(exclude)
    if not ordered:
        if not isinstance(include, set):
            include = set(include)
        if not isinstance(exclude, set):
            exclude = set(exclude)
        sel = []
        for k, name in enumerate(ch_names):
            if (len(include) == 0 or name in include) and name not in exclude:
                sel.append(k)
    else:
        if not isinstance(include, list):
            include = list(include)
        if len(include) == 0:
            include = list(ch_names)
        if not isinstance(exclude, list):
            exclude = list(exclude)
        sel, missing = list(), list()
        for name in include:
            if name in ch_names:
                if name not in exclude:
                    sel.append(ch_names.index(name))
            else:
                missing.append(name)
        if len(missing):
            raise ValueError('Missing channels from ch_names required by '
                             'include:\n%s' % (missing,))
    return np.array(sel, int)