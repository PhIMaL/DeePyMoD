""" Constains a tool to convert from Tensorboard to Pandas DataFrame """

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from natsort import natsorted
import matplotlib.pyplot as plt


def load_tensorboard(path: str) -> pd.DataFrame:
    """Loads tensorboard files into a pandas dataframe. Assumes one run per folder!

    Args:
        path (string): path of folder with tensorboard files.

    Returns:
        DataFrame: Pandas dataframe with all run data.
    """

    event_paths = [
        file
        for file in os.walk(path, topdown=True)
        if file[2][0][: len("events")] == "events"
    ]

    df = pd.DataFrame()
    steps = None  # steps are the same for all files

    for event_idx, path in enumerate(event_paths):
        summary_iterator = EventAccumulator(os.path.join(path[0], path[2][0])).Reload()
        tags = summary_iterator.Tags()["scalars"]
        data = [
            [event.value for event in summary_iterator.Scalars(tag)] for tag in tags
        ]
        if steps is None:
            steps = [event.step for event in summary_iterator.Scalars(tags[0])]

        # Adding to dataframe
        tags = [tag.replace("/", "_") for tag in tags]  # for name consistency
        if (
            event_idx > 0
        ):  # We have one file in the top level, so after we need to use folder name
            tags = [path[0].split("/")[-1]]

        for idx, tag in enumerate(tags):
            try:
                df[tag] = data[idx]
            except ValueError:  # more debugging info
                print(
                    f"Warning: Either the {tag = } of `df` or {idx = } of `data` do not exist! Check for pre-existing saved files. "
                )
        df.index = steps
    return df


def plot_history(foldername: str):
    """Plots the training history of the model."""
    history = load_tensorboard(foldername)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for history_key in history.keys():
        history_key_parts = history_key.split("_")
        if history_key_parts[0] == "loss":
            if history_key_parts[-1] == "0":
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle="--",
                )
            elif history_key_parts[-1] == "1":
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle=":",
                )
            else:
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle="-",
                )
            if history_key_parts[0] == "remaining":
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1]
                    + "_"
                    + history_key_parts[3]
                    + "_"
                    + history_key_parts[4],
                    linestyle="-.",
                )
        if history_key_parts[0] == "coeffs":
            if history_key_parts[2] == "0":
                axs[1].plot(
                    history[history_key],
                    label=history_key_parts[2]
                    + "_"
                    + history_key_parts[3]
                    + "_"
                    + history_key_parts[4],
                    linestyle="--",
                )
            elif history_key_parts[2] == "1":
                axs[1].plot(
                    history[history_key],
                    label=history_key_parts[2]
                    + "_"
                    + history_key_parts[3]
                    + "_"
                    + history_key_parts[4],
                    linestyle=":",
                )
            else:
                axs[1].plot(
                    history[history_key],
                    label=history_key_parts[2]
                    + "_"
                    + history_key_parts[3]
                    + "_"
                    + history_key_parts[4],
                    linestyle="-",
                )
        if history_key_parts[0] == "unscaled":
            if history_key_parts[3] == "0":
                axs[2].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="--",
                )
            elif history_key_parts[3] == "1":
                axs[2].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle=":",
                )
            else:
                axs[2].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="-",
                )
        if history_key_parts[0] == "estimator":
            if history_key_parts[3] == "0":
                axs[3].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="--",
                )
            elif history_key_parts[3] == "1":
                axs[3].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle=":",
                )
            else:
                axs[3].plot(
                    history[history_key],
                    label=history_key_parts[3]
                    + "_"
                    + history_key_parts[4]
                    + "_"
                    + history_key_parts[5],
                    linestyle="-",
                )

    # axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[2].set_ylim([-2, 2])
    axs[3].set_ylim([-2, 2])

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.show()
