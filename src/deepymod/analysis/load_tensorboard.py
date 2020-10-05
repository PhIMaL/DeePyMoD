import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from natsort import natsorted
import matplotlib.pyplot as plt


def load_tensorboard(path):
    '''Function to load tensorboard file from a folder.
    Assumes one file per folder!'''

    event_paths = [file for file in os.walk(path, topdown=True) if file[2][0][:len('events')] == 'events']

    df = pd.DataFrame()
    steps = None # steps are the same for all files

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


class Results:
    def __init__(self, path):
        self.df = load_tensorboard(path)
        self.keys = self.get_keys()

    def get_keys(self):
        mse_keys = natsorted([key for key in self.df.keys() if key[:len('loss_mse')] == 'loss_mse'])
        reg_keys = natsorted([key for key in self.df.keys() if key[:len('loss_reg')] == 'loss_reg'])
        l1_keys = natsorted([key for key in self.df.keys() if key[:len('loss_l1')] == 'loss_l1'])
        loss_key = 'loss_loss'

        coeff_keys = natsorted([key for key in self.df.keys() if key[:len('coeffs')] == 'coeffs'])
        unscaled_coeff_keys = natsorted([key for key in self.df.keys() if key[:len('unscaled')] == 'unscaled'])
        estimator_coeff_keys = natsorted([key for key in self.df.keys() if key[:len('estimator')] == 'estimator'])

        return {'mse': mse_keys, 'reg': reg_keys, 'l1': l1_keys, 'loss': loss_key, 
                'coeffs': coeff_keys, 'unscaled_coeffs': unscaled_coeff_keys, 'estimator_coeffs': estimator_coeff_keys}

    def plot_losses(self):
        fig, axes = plt.subplots(figsize=(12, 7), nrows=2, ncols=2, tight_layout=True)

        ax = axes[0, 0]
        ax.plot(self.df.index, self.df[self.keys['loss']])
        ax.set_title('Loss')

        ax = axes[0, 1]
        for key in self.keys['mse']:
            ax.semilogy(self.df.index, self.df[key], label=key[9:])
        ax.set_title('MSE')
        ax.legend()
        ax.set_xlabel('Epoch')

        ax = axes[1, 0]
        for key in self.keys['reg']:
            ax.semilogy(self.df.index, self.df[key], label=key[9:])
        ax.set_title('MSE')
        ax.legend()
        ax.set_xlabel('Epoch')

        ax = axes[1, 1]
        for key in self.keys['l1']:
            ax.semilogy(self.df.index, self.df[key], label=key[8:])
        ax.set_title('MSE')
        ax.legend()
        ax.set_xlabel('Epoch')

    def plot_coeffs(self):
        fig, axes = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, tight_layout=True)

        ax = axes[0]
        for key in self.keys['coeffs']:
            ax.plot(self.df.index[1:], self.df[key][1:], label=key[7:])
        ax.set_title('Coeffs')
        ax.legend(ncol=2)
        ax.set_xlabel('Epoch')

        ax = axes[1]
        for key in self.keys['unscaled_coeffs']:
            ax.plot(self.df.index[1:], self.df[key][1:], label=key[16:])
        ax.legend(ncol=2)
        ax.set_title('Unscaled coeffs')
        ax.set_xlabel('Epoch')

        ax = axes[2]
        for key in self.keys['estimator_coeffs']:
            ax.plot(self.df.index[1:], self.df[key][1:], label=key[17:])
        ax.legend(ncol=2)
        ax.set_title('Estimator coeffs')
        ax.set_xlabel('Epoch')