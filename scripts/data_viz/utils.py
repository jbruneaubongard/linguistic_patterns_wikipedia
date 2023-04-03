import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import colorsys


to_timestamp = np.vectorize(lambda x: x.timestamp())

def get_intervals(start_date: str, end_date: str, duration_in_days: int):
    """
    Creates pandas intervals of length duration_in_days between start_date and end_date and associated bins
    start_date and end_date are strings of the form 'YYYY-MM-DD'
    """

    intervals = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if start > end:
        return intervals
    curr = start
    bins = [curr]
    while curr <= end:
        intervals.append(pd.Interval(pd.Timestamp(curr), pd.Timestamp(curr + timedelta(days=duration_in_days)), closed='left'))
        curr += timedelta(days=duration_in_days)
        bins.append(curr)
    return np.array(intervals), to_timestamp(bins)

def add_interval_column(df: pd.DataFrame, start_date: str, end_date: str, duration_in_days: int):
    """
    Adds a column to df with the corresponding interval of length duration_in_days between start_date and end_date
    start_date and end_date are strings of the form 'YYYY-MM-DD'
    """

    _, bins = get_intervals(start_date, end_date, duration_in_days)
    df_copy = df.copy()
    df_copy.loc[:, f'interval_{duration_in_days}'] = pd.cut(df_copy['timestamp'], bins=bins)
    return df_copy
    
def timestamp_interval_to_string(interval: pd.Interval):
    """
    Converts a pandas interval to a string of the form '(YYYY-MM-DD, YYYY-MM-DD]'
    """

    left = pd.to_datetime(interval.left, unit='s').strftime("%Y-%m-%d")
    right = pd.to_datetime(interval.right, unit='s').strftime("%Y-%m-%d")
    return f"({left}, {right}]"

def get_list_speakers_per_interval(speakers_per_interval: list, threshold=1):
    """
    Returns a list of intervals and a list of sets of speakers for each interval, where each speaker must have written at least threshold messages in the interval
    """
    sets_of_speakers = []
    intervals = []
    for interval in speakers_per_interval.index:
        intervals.append(interval)
        speakers = set(speakers_per_interval.loc[interval])
        speakers_threshold = {speaker for speaker in speakers if speakers_per_interval.loc[interval].count(speaker) >= threshold}
        sets_of_speakers.append(speakers_threshold)
    return intervals, sets_of_speakers

def get_overlap_speakers_for_plot(sets_of_speakers: list):
    """
    Returns a list of lists of sets of speakers for each interval
    overlap_speakers[i][j] is the set of speakers in the ith interval that appeared for the first time in the jth interval
    /!\ overlap_speakers[i][j] contains only speakers that appeared in all of the intervals from j to i
    """
    overlap_speakers = []
    for i in range(len(sets_of_speakers)):
        if i==0:
            overlap_speakers.append([sets_of_speakers[i]])
        else:
            list_of_sets = [overlap_speakers[i-1][j].intersection(sets_of_speakers[i]) for j in range(i)]
            list_of_sets.append(sets_of_speakers[i].difference(set.union(*list_of_sets)))
            overlap_speakers.append(list_of_sets)
    return overlap_speakers

def get_color_range(nb_colors: int):
    """
    Returns a rainbow list of nb_colors colors
    """
    color_list = []
    for k in range(nb_colors):
      r,g,b = colorsys.hsv_to_rgb(k / nb_colors * 1.0, 0.5, 0.8)
      color_list.append(f'rgb({round(255*r)},{round(255*g)},{round(255*b)})')
    return color_list

def get_bar_plot(threshold: int, speakers_per_interval:list, color_list: list):
    """
    Returns the bar plot, base of sankey diagram
    Each speaker must have written at least threshold messages in an interval to be considered
    """

    # Get data 
    intervals, sets_speakers = get_list_speakers_per_interval(speakers_per_interval, threshold = threshold)
    df_plot = pd.DataFrame([[len(li[i]) for i in range(len(li))] for li in get_overlap_speakers_for_plot(sets_speakers)], index=intervals, columns=intervals)

    # Core bar chart
    px_fig = px.bar(
        df_plot.reset_index(), 
        x="index", 
        y=df_plot.columns,
        )

    # Set color and pattern
    for j,barchart in enumerate(px_fig['data']):
        shape = '.' if j%2 == 0  else '+'
        barchart.marker = {'color': color_list[j], 'pattern': {'shape': shape, 'size' : 10}}

    return px_fig

def get_button(idx_threshold: int, threshold: int, nb_traces_per_plot: int, duration_in_days: int):
    """
    Returns a button for the sankey diagram corresponding to the threshold
    """
    visible = [True if i in range(idx_threshold*nb_traces_per_plot, (idx_threshold + 1)*nb_traces_per_plot) else False for i in range(nb_traces_per_plot**2)]

    button = dict(
        label=f'Threshold = {threshold}',
        method='update',
        args=[
            {'visible': visible},
            {'title': f'Threshold per interval = {threshold}, intervals of length {duration_in_days} days'}
        ]
    )
    return button

def get_sankey_plot(df: pd.DataFrame, start_date: str, end_date: str, duration_in_days: int, list_thresholds: list, log_scale=False):
    """
    Returns the complete sankey diagram with different thresholds accessible through buttons
    df is the corpus dataframe
    start_date and end_date are strings of the form 'YYYY-MM-DD'
    duration_in_days is the length of the intervals in days
    list_thresholds is a list of thresholds (minimum number of messages per speaker per interval)
    log_scale is a boolean, if True, the y-axis is in log scale
    """

    df = add_interval_column(df, start_date, end_date, duration_in_days)
    df[f'interval_{duration_in_days}_str'] = df[f'interval_{duration_in_days}'].apply(timestamp_interval_to_string)
    speakers_per_interval = df.groupby(by=f'interval_{duration_in_days}_str')['speaker'].apply(list)

    # Get range of colors
    color_list = get_color_range(len(speakers_per_interval))

    # Initialize figure
    fig = go.Figure()
    buttons = []

    # Create bar plots for each threshold
    for idx_threshold,threshold in enumerate(list_thresholds):
      px_fig = get_bar_plot(threshold, speakers_per_interval, color_list)

      # Get max y_axis value
      if idx_threshold==0:
        max_value = np.nanmax([np.nansum([trace['y'][i] for trace in px_fig.data]) for i in range(len(px_fig.data[0]['y']))])

      fig.add_traces(px_fig.data)
      buttons.append(get_button(idx_threshold, threshold, len(speakers_per_interval), duration_in_days))

    # Layout
    fig.update_layout(
        title_text=f"Diagrams for different thresholds, intervals of length {duration_in_days} days",
        barmode='stack',
        height = 700,
        width = 1800,
        yaxis=dict(range=[0,max_value + 100]),
        updatemenus=[
            dict(
              type="buttons",
              direction="right",
              x=1,
              y=1.2,
              showactive=False,
              buttons=buttons
       )
      ]
    )

    if log_scale:
      fig.update_yaxes(
          type='log', 
          range=[0,np.log10(max_value + 100)]
          )

    fig.show()
    
    return None

def get_histogram(df_admin, df_non_admin, start_date, end_date, bin_size_months):
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Declare the bins

    xbins=dict(
            start=start_date,
            end=end_date,
            size=f'M{bin_size_months}' # bin_size_months bin size
            )

    # Plot the histograms

    trace_admin = go.Histogram(
        x=df_admin['date'], 
        name='admin', 
        xbins=xbins, 
        autobinx=False,
        marker_color='mediumvioletred'
    )

    trace_non_admin = go.Histogram(
        x=df_non_admin['date'], 
        name='non-admin', 
        xbins=xbins, 
        autobinx=False,
        marker_color='blueviolet'
    )

    fig.add_trace(trace_admin)
    fig.add_trace(trace_non_admin)

    # Compute and plot the percentage of admin utterances

    intervals, np_bins = get_intervals(start_date, end_date, 92)
    bin_centers = np.array([bin.mid for bin in intervals])
    admin_counts, _ = np.histogram(df_admin['timestamp'], bins=np_bins)
    non_admin_counts, _ = np.histogram(df_non_admin['timestamp'], bins=np_bins)
    ratios = 100 * admin_counts / (non_admin_counts + admin_counts)

    trace_ratio = go.Scatter(
        x=bin_centers,
        y=ratios,
        name='Percentage of admin utterances',
        line=dict(color='black', width=2),
        mode='lines'
    )

    fig.add_trace(trace_ratio, secondary_y=True)

    fig.update_layout(
        bargap=0.05,
        barmode= 'stack',
        title_text='Histogram of utterances over time, depending on the status of the writer', # title of plot
        xaxis_title_text='Date', # xaxis label
        )

    fig.update_yaxes(title_text="Number of utterances", secondary_y=False)
    fig.update_yaxes(title_text="Percentage of admin utterances", secondary_y=True)

    fig.show()

    return None
