import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import webbrowser
import pickle

class ClusterVisualization:
    def __init__(self, plot1, plot2_dict):
        self.plot1 = plot1
        self.plot2_dict = plot2_dict
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        # Create the layout for the Dash app
        self.app.layout = html.Div([
            dcc.Graph(
                id='plot1',
                figure=self.plot1,
                config={'displayModeBar': True}  # Enable the mode bar
            ),
            dcc.Graph(
                id='hovered-plot',  # This will display the plot corresponding to the hovered node
                config={'displayModeBar': True}
            ),
            html.Div(id='hover-output')
        ])
    
    def setup_callbacks(self):
        # Callback to handle hover events and update the hovered plot
        @self.app.callback(
            [dash.dependencies.Output('hovered-plot', 'figure'),
             dash.dependencies.Output('hover-output', 'children')],
            [dash.dependencies.Input('plot1', 'hoverData')]
        )
        def display_hover_data(hoverData):
            if hoverData:
                # Extract the hovered node name
                node_name = int(hoverData['points'][0]['text'])
                
                # Get the corresponding plot from the dictionary
                hovered_plot = self.plot2_dict.get(node_name, go.Figure())  # Default to empty plot if not found

                return hovered_plot, f'Hovering over: {node_name}'
            
            return go.Figure(), "Hover over a node!"
    
    def run(self):
        # Path to the Chrome executable
        chrome_path = '"C:/Program Files/Google/Chrome/Application/chrome.exe"'  # Update this path if necessary

        # Open the app in Chrome
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
        webbrowser.get('chrome').open('http://127.0.0.1:8050/')

        # Run the Dash app
        print(f"running on {'http://127.0.0.1:8050/'}")
        self.app.run_server(debug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass plot data through command line.')
    parser.add_argument('--cluster_plot_dir', type=str, required=True, help='JSON string of cluster plot data')
    parser.add_argument('--graph_plots_dir', type=str, required=True, help='JSON string of graph plots data')

    args = parser.parse_args()

    # Process command-line arguments
    cluster_plot_str = args.cluster_plot_dir  # This should be a JSON string
    graph_plots_str = args.graph_plots_dir  # This should be a JSON string

    with open(f'{cluster_plot_str}.pkl', 'rb') as file: cluster_plot = pickle.load(file)
    with open(f'{graph_plots_str}.pkl', 'rb') as file: graph_plots = pickle.load(file)

    # Create and run the ClusterVisualization
    plot = ClusterVisualization(cluster_plot, graph_plots)
    plot.run()