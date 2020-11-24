""" Constains a tool to convert from Tensorboard to Pandas DataFrame """

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from natsort import natsorted
import matplotlib.pyplot as plt


def load_tensorboard(path: str) -> pd.DataFrame:
    """ Loads tensorboard files into a pandas dataframe. Assumes one run per folder!

    Args:
        path (string): path of folder with tensorboard files.

    Returns:
        DataFrame: Pandas dataframe with all run data.
    """

    event_paths = [file for file in os.walk(path, topdown=True) if file[2][0][:len('events')] == 'events']

    df = pd.DataFrame()
    steps = None  # steps are the same for all files

    for event_idx, path in enumerate(event_paths):
        summary_iterator = EventAccumulator(os.path.join(path[0], path[2][0])).Reload()
        tags = summary_iterator.Tags()['scalars']
        data = [[event.value for event in summary_iterator.Scalars(tag)] for tag in tags]
        if steps is None:
            steps = [event.step for event in summary_iterator.Scalars(tags[0])]
        
        # Adding to dataframe
        tags = [tag.replace('/', '_') for tag in tags] # for name consistency
        if event_idx > 0: # We have one file in the top level, so after we need to use folder name
            tags = [path[0].split('/')[-1]]
        
        for idx, tag in enumerate(tags):
            df[tag] = data[idx]
        df.index = steps
    return df