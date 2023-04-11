import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from utils import get_color_range, get_button, timestamp_interval_to_string, dt64_to_timestamp


class UttDataFrame(pd.DataFrame):
    """
    A useful class to manipulate the wikiconv corpus
    """
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self['timestamp'] = self['timestamp'].apply(lambda x: float(x))

      # Convert timestamp to datetime
      self['date'] = pd.to_datetime(self['timestamp'], unit='s')
      
    def admin(self):
       """
       Returns a dataframe with only the utterances of admin speakers (at the time of the utterance)
       """
       return self[self['meta.is-admin'].apply(lambda x: x)]
   
    def non_admin(self):
       """
       Returns a dataframe with only the utterances of non-admin speakers (at the time of the utterance)
       """
       return self[self['meta.is-admin'].apply(lambda x: not x)]
    

    def get_histogram(
          self, 
          start_date: pd.Timestamp, 
          end_date: pd.Timestamp, 
          bin_size_in_days: int
    ):
        """
        Returns a plotly figure with histograms of the number of utterances per bin_size_in_days for admin and non-admin users,
        and a line plot of the percentage of admin utterances per bin_size_in_days
        start_date and end_date must be in pandas Timestamp format (use pd.to_datetime(str))
        """

        # Initialize the figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Declare the bins
        xbins=dict(
                start=start_date,
                end=end_date,
                size=86400000*bin_size_in_days # Size must be in milliseconds
                )

        # Get the histogram traces
        trace_admin = go.Histogram(
            x=self.admin()['date'], 
            name='admin', 
            xbins=xbins, 
            autobinx=False,
            marker_color='mediumvioletred'
        )

        trace_non_admin = go.Histogram(
            x=self.non_admin()['date'], 
            name='non-admin', 
            xbins=xbins, 
            autobinx=False,
            marker_color='blueviolet'
        )

        fig.add_trace(trace_admin)
        fig.add_trace(trace_non_admin)

        # Get the trace for the percentage of admin utterances
        bins = pd.interval_range(start_date, end_date + timedelta(days=bin_size_in_days),freq=f'{bin_size_in_days}D', closed='left')
        np_bins = dt64_to_timestamp(np.array(pd.date_range(start_date, end_date + timedelta(days=bin_size_in_days), freq=f'{bin_size_in_days}D')))
        bin_centers = np.array([bin.mid for bin in bins])
        
        admin_counts, _ = np.histogram(self.admin()['timestamp'], bins=np_bins)
        non_admin_counts, _ = np.histogram(self.non_admin()['timestamp'], bins=np_bins)
        ratios = 100 * admin_counts / (non_admin_counts + admin_counts)
        ratios= np.nan_to_num(ratios, copy=False)

        trace_ratio = go.Scatter(
            x=bin_centers,
            y=ratios,
            name='Percentage of admin utterances',
            line=dict(color='black', width=2),
            mode='lines'
        )

        fig.add_trace(trace_ratio, secondary_y=True)

        # Update the layout
        fig.update_layout(
            bargap=0.05,
            barmode= 'stack',
            title_text=f'Histogram of utterances over time, depending on the status of the writer (bins size = {bin_size_in_days} days)',
            xaxis_title_text='Date', 
            )

        fig.update_yaxes(title_text="Number of utterances", secondary_y=False)
        fig.update_yaxes(title_text="Percentage of admin utterances", secondary_y=True)

        fig.show()

        return fig
    
    def filter_by_activity(
          self,
          start_date: pd.Timestamp,
          end_date: pd.Timestamp,
          duration_in_days: int,
          threshold: int
    ):
        """
        Returns a dataframe with only the utterances of speakers who have at least threshold utterances \
        in each interval of duration_in_days days, between start_date and end_date
        """
        # Add interval column stating which interval of duration_in_days days \
        # each utterance belongs to (if doesn't exist yet)
        if f'interval_{duration_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}' not in self.columns:
           intervals = pd.interval_range(start_date, end_date + timedelta(days=duration_in_days),freq=f'{duration_in_days}D', closed='left')
           self[f'interval_{duration_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}'] = pd.cut(self['date'], bins=intervals).apply(timestamp_interval_to_string)

        # Group the dataframe by speaker and interval
        grouped = self.groupby(['speaker', f'interval_{duration_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}']).count()

        # Get speakers who wrote more than threshold messages per interval
        filtered_speakers = grouped[grouped['timestamp'] >= threshold].reset_index()['speaker'].unique()

        # Return a filtered dataframe containing only the selected speakers
        return UttDataFrame(self[self['speaker'].isin(filtered_speakers)])
    
    def get_overlapping_speakers(
            self,
            speakers_per_bin: list,	
    ):
        """
        Returns a list of lists of sets of speakers for each bin, such that:
        overlap_speakers[i][j] is the set of speakers in the ith bin that appeared for the first time in the jth bin
        /!\ overlap_speakers[i][j] contains only speakers that appeared in all of the bins from j to i
        """
        overlap_speakers = []
        for i in range(len(speakers_per_bin)):
            if i==0:
                overlap_speakers.append([speakers_per_bin[i]])
            else:
                list_of_sets = [overlap_speakers[i-1][j].intersection(speakers_per_bin[i]) for j in range(i)]
                list_of_sets.append(speakers_per_bin[i].difference(set.union(*list_of_sets)))
                overlap_speakers.append(list_of_sets)
        return overlap_speakers
    
    def get_bar_plot(
            self,
            bins: list,
            speakers_per_bin: list,
            color_list: list,
    ):
        df_plot = pd.DataFrame(
            [[len(li[i]) for i in range(len(li))] for li in self.get_overlapping_speakers(speakers_per_bin)],
            index=[timestamp_interval_to_string(bin) for bin in bins], 
            columns=[timestamp_interval_to_string(bin) for bin in bins],
        )

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

    def get_flow_plot(
          self, 
          start_date: pd.Timestamp, 
          end_date: pd.Timestamp, 
          duration_in_days_criterion: int, 
          list_thresholds: list, 
          bin_size_in_days: int,
          log_scale: bool = False
    ):
        """
        duration_in_days_criteria: the duration of the intervals used to filter the dataframe
        """
        # Get bins
        bins = pd.interval_range(start_date, end_date + timedelta(days=bin_size_in_days),freq=f'{bin_size_in_days}D', closed='left')
        
        # Get range of colors
        color_list = get_color_range(len(bins))

        # Initialize figure
        fig = go.Figure()
        buttons = []

        # Create bar plots for each threshold
        for idx_threshold,threshold in enumerate(list_thresholds):
            # Get filtered dataframe 
            filtered_df = self.filter_by_activity(start_date, end_date, duration_in_days_criterion, threshold)

            # Get sets of speakers per bin
            if f'interval_{bin_size_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}' not in filtered_df.columns:
                filtered_df[f'interval_{bin_size_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}'] = pd.cut(filtered_df['date'], bins=bins).apply(timestamp_interval_to_string)
            speakers_per_bin = filtered_df.groupby(by=f'interval_{bin_size_in_days}D_start={str(start_date)[:10]}_end={str(end_date)[:10]}')['speaker'].apply(set).to_list()
            
            # Get bar plot
            px_fig = filtered_df.get_bar_plot(bins, speakers_per_bin, color_list)

            # Get max y_axis value
            if idx_threshold==0:
                max_value = np.nanmax([np.nansum([trace['y'][i] for trace in px_fig.data]) for i in range(len(px_fig.data[0]['y']))])
            fig.add_traces(px_fig.data)
            buttons.append(get_button(idx_threshold, threshold, len(bins), duration_in_days_criterion, bin_size_in_days))

        # Layout
        fig.update_layout(
            title_text=f"Diagrams for different activity criteria, intervals of length {bin_size_in_days} days",
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
    
    def assign_reply_to_speaker(self):
        """
        Assigns a reply_to_speaker column to the dataframe, which contains the speaker of the message the utterance is replying to
        """
        self['reply_to_speaker'] = None
        for idx in self[self["reply_to"] != None].index:
            if self.loc[idx,'reply_to'] in self.index:
                self.loc[idx,'reply_to_speaker'] = self.loc[self.loc[idx,'reply_to'],'speaker']
        return UttDataFrame(self)
