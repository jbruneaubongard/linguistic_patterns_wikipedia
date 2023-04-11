import numpy as np
import pandas as pd
import colorsys

dt64_to_timestamp = np.vectorize(lambda x: pd.Timestamp(x).timestamp())

def timestamp_interval_to_string(interval: pd.Interval):
    """
    Converts a pandas interval to a string of the form '(YYYY-MM-DD, YYYY-MM-DD]'
    """

    left = pd.to_datetime(interval.left, unit='s').strftime("%Y-%m-%d")
    right = pd.to_datetime(interval.right, unit='s').strftime("%Y-%m-%d")
    return f"({left}, {right}]"

def get_color_range(nb_colors: int):
    """
    Returns a rainbow list of nb_colors colors
    """
    color_list = []
    for k in range(nb_colors):
      r,g,b = colorsys.hsv_to_rgb(k / nb_colors * 1.0, 0.5, 0.8)
      color_list.append(f'rgb({round(255*r)},{round(255*g)},{round(255*b)})')
    return color_list

def get_button(idx_threshold: int, threshold: int, nb_traces_per_plot: int, duration_in_days_criterion: int, bin_size_in_days: int):
    """
    Returns a button for the sankey diagram corresponding to the threshold
    """
    visible = [True if i in range(idx_threshold*nb_traces_per_plot, (idx_threshold + 1)*nb_traces_per_plot) else False for i in range(nb_traces_per_plot**2)]

    button = dict(
        label=f'Threshold = {threshold}',
        method='update',
        args=[
            {'visible': visible},
            {'title': f'Threshold per interval of length {duration_in_days_criterion} days = {threshold}, bins of size {bin_size_in_days} days'}
        ]
    )
    return button
