""" Constains a tool to convert from Tensorboard to Pandas DataFrame """

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os


def load_tensorboard(path: str) -> pd.DataFrame:
    """Loads tensorboard files into a pandas dataframe. Assumes one run per folder!

    Args:
        path (string): path of folder with tensorboard files.

    Returns:
        DataFrame: Pandas dataframe with all run data.
    """

    def event_filter(x):
        is_event = False
        if len(x[2]) > 0:  # check if folder contains files
            if x[2][0].find("event") != -1:  # check if folder contains eventfile
                is_event = True
        return is_event

    df = pd.DataFrame()
    steps = None  # steps are the same for all files

    for path in filter(event_filter, os.walk(path, topdown=True)):
        summary_iterator = EventAccumulator(os.path.join(path[0], path[2][0])).Reload()
        tags = summary_iterator.Tags()["scalars"]
        data = [
            [event.value for event in summary_iterator.Scalars(tag)] for tag in tags
        ]
        if steps is None:
            steps = [event.step for event in summary_iterator.Scalars(tags[0])]

        # Adding to dataframe
        tags = [tag.replace("/", "_") for tag in tags]  # for name consistency
        if len(path[1]) == 0:  # if there's no deeper folder, add the folder name.
            tags = [path[0].split("/")[-1]]

        for idx, tag in enumerate(tags):
            df[tag] = data[idx]
        df.index = steps
    return df
