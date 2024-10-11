import plotly.graph_objects as go
from typing import Dict,List,Union
from llama_index.core.schema import TextNode
from plotly.subplots import make_subplots

def plot_cluster_counts(contexts:Dict[str,TextNode], 
                        graph_name:str, 
                        x_label:str, 
                        y_label:str):
    
    return (_plot(x_values   = list(contexts.keys()),
                  y_values   = [len(value) for value in contexts.values()],
                  graph_name = graph_name,
                  x_label    = x_label,
                  y_label    = y_label))
            
def plot_avg_cluster_variance(avg_variance_dict: Dict[int,List[float]],
                            graph_name:str, 
                            x_label:str, 
                            y_label:str):
    
    return (_plot(x_values   = list(avg_variance_dict.keys()),
                  y_values   = [value for value in avg_variance_dict.values()],
                  graph_name = graph_name,
                  x_label    = x_label,
                  y_label    = y_label))

def _plot(x_values:List[int],
          y_values:List[Union[int,float]], 
          graph_name:str, 
          x_label:str, 
          y_label:str):
    
    # Creating the bar graph
    fig = go.Figure(data=[
        go.Bar(x=x_values, y=y_values)
    ])
    
    # Setting title and axis labels
    fig.update_layout(
        title=graph_name,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis={
            'tickangle': -45,  # Optional: Rotate x-axis labels for better readability
            'automargin': True,
            'range': [0, 100],  # Initial visible range
        },
        yaxis={
            'automargin': True
        },
        width=1600,  # Set the width of the figure
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    
    # Adding scroll functionality
    #fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def plot_mean_avg_cluster_similarity(mean_avg_cluster_sim_dict: Dict[int,List[float]],
                                     graph_name:str, 
                                     x_label:str, 
                                     y_label:str):
    
    return (_plot(x_values   = list(mean_avg_cluster_sim_dict.keys()),
                y_values     = [value for value in mean_avg_cluster_sim_dict.values()],
                graph_name   = graph_name,
                x_label      = x_label,
                y_label      = y_label))


def plot_graphs(figures):
    # Determine the number of rows needed
    num_rows = len(figures)

    # Create a subplot with the specified number of rows and 1 column
    fig = make_subplots(rows=num_rows, cols=1, vertical_spacing=0.1, shared_xaxes=True)

    # Add each figure to the subplot
    for i, figure in enumerate(figures):
        for trace in figure.data:
            fig.add_trace(trace, row=i+1, col=1)

        # Transfer layout properties from the individual figure to the subplot
        fig.update_xaxes(title_text=figure.layout.xaxis.title.text, row=i+1, col=1)
        fig.update_yaxes(title_text=figure.layout.yaxis.title.text, row=i+1, col=1)
        if figure.layout.title:
            fig.add_annotation(
                text=figure.layout.title.text,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=1.1,
                showarrow=False,
                row=i+1,
                col=1
            )

    # Add rangeslider only for the last plot
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor="gold", thickness=0.1), 
        row=num_rows, col=1
    )

    # Update the layout to adjust the height of the subplots
    fig.update_layout(height=300 * num_rows, showlegend=False)

    # Show the combined figure
    fig.show()