import plotly.graph_objects as go

def plot_cluster_counts(data_dict, graph_name, x_label, y_label):
    # Extracting keys and their corresponding list lengths
    x_values = list(data_dict.keys())
    y_values = [len(value) for value in data_dict.values()]
    
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
        width=1200,  # Set the width of the figure
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    
    # Adding scroll functionality
    fig.update_xaxes(rangeslider_visible=True)
    
    # Display the graph
    fig.show()