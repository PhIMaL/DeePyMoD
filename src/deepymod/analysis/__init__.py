"""
Analysis library 
================

Tools to interpert the results of DeePyMoD.

Tools
-----

    load_tensorboard    convert the tensorboard files into a Pandas DataFrame.
    plot_history        plot the training history of the model.

"""
from .load_tensorboard import load_tensorboard
from .load_tensorboard import plot_history
