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
                id='hovered-plot',  # This will display the plot corresponding to the hovered node or input
                config={'displayModeBar': True}
            ),
            html.Div(id='hover-output'),
            dcc.Input(id='cluster-id-input', type='number', placeholder='Enter cluster ID', debounce=True),
            html.Button('Submit', id='submit-button', n_clicks=0),
            html.Div(id='input-output')
        ])
    
    def setup_callbacks(self):
        # Callback to handle both hover and input events
        @self.app.callback(
            [dash.dependencies.Output('hovered-plot', 'figure'),
             dash.dependencies.Output('hover-output', 'children'),
             dash.dependencies.Output('input-output', 'children')],
            [dash.dependencies.Input('plot1', 'hoverData'),
             dash.dependencies.Input('submit-button', 'n_clicks')],
            [dash.dependencies.State('cluster-id-input', 'value')]
        )
        def display_hover_or_input(hoverData, n_clicks, cluster_id):
            ctx = dash.callback_context
            
            # Check which input triggered the callback (hover or submit)
            if not ctx.triggered:
                return go.Figure(), "Hover over a node or enter a cluster ID!", ""

            # Check if hover data triggered the callback
            if 'plot1.hoverData' in ctx.triggered[0]['prop_id'] and hoverData:
                node_name = int(hoverData['points'][0]['text'])
                hovered_plot = self.plot2_dict.get(node_name, go.Figure())  # Get plot based on hovered node
                return hovered_plot, f'Hovering over: {node_name}', ""

            # Check if submit button triggered the callback and cluster_id is valid
            if 'submit-button.n_clicks' in ctx.triggered[0]['prop_id'] and cluster_id is not None:
                cluster_id = int(cluster_id)
                input_plot = self.plot2_dict.get(cluster_id, go.Figure())  # Get plot based on input cluster ID
                return input_plot, "", f'Selected cluster ID: {cluster_id}'
            
            return go.Figure(), "Hover over a node or enter a cluster ID!", ""

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