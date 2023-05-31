import matplotlib.pyplot as plt  # for plotting
from IPython.display import HTML
from plotnine import *
import pandas as pd
import numpy as np


def error_plot(train_error, test_error, lengths: list):
    # plot the training error vs. the sample size  n  shown on a log-scale
    ggplot(pd.DataFrame({'x': np.concatenate((np.log(lengths), np.log(lengths))),
                         'y': train_error + test_error,
                         'set': ['train'] * 15 + ['test'] * 15}), aes(x='x', y='y', color='set')) + \
    geom_point() + \
    geom_line() + \
    labs(title="Error Plot by Samples size", x="Sample Size[log]", y="Error")


def plot_loss_per_epochs(epochs: int, epochLoss: list, label: str):
    plt.plot(range(0, epochs), epochLoss, label=label)
    plt.axvline([i for i in range(epochs) if epochLoss[i] == max(epochLoss)], c="red", ls='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
