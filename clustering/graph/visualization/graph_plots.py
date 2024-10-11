import networkx as nx
import plotly.graph_objects as go
from typing import Dict
import textwrap

def create_networkx_plot(G:nx.Graph, contexts):
    # Generate positions for nodes using the spring layout
    pos = nx.spring_layout(G,k=0.25)

    # Create edge traces
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = round(edge[2].get('weight', 1), 3)
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                text=f'Weight: {weight}',
                mode='lines'
            )
        )

    # Initialize node trace with empty lists for x, y, and text
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers',
        hoverinfo='text',
        text=[],  # Initialize as an empty list
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
        )
    )

    node_adjacencies = []
    node_texts = []  # Initialize a separate list for node texts
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

        # Number of connections (degree)
        node_adjacencies.append(len(G.adj[node]))
        adj_nodes = G.adj[node]
        node_text = f"{node} : "+"<br>".join(textwrap.wrap(f'{contexts[str(node)].text}', width=100))
        node_texts.append(node_text)  # Add to node_texts list

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_texts  # Set the text attribute

    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='<br>Network graph visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Graph Visualization with Plotly",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

    # Save the figure as an HTML file
    #pio.write_html(fig, file=filename, auto_open=True)

def get_individual_cluster_graph_plots(G:Dict[int, nx.Graph],contexts):
    plots = {}
    for key in G:
        plots[key] = create_networkx_plot(G[key],contexts)
    
    return plots

def get_connected_graph_plot(graph, clusters):
    
    cluster_graph = nx.Graph()
    
    # Add each cluster as a single node in the new graph
    cluster_labels = list(clusters.keys())
    cluster_graph.add_nodes_from(cluster_labels)
    
    # Add edges between clusters based on the existing fully connected graph
    for (node1, node2) in graph.edges():
        # Find the clusters that each node belongs to
        cluster1 = next((label for label, items in clusters.items() if node1 in items), None)
        cluster2 = next((label for label, items in clusters.items() if node2 in items), None)
        
        # If both nodes belong to different clusters, add an edge between those clusters
        if cluster1 is not None and cluster2 is not None and cluster1 != cluster2:
            cluster_graph.add_edge(cluster1, cluster2)

    # Get positions for nodes in the cluster graph using spring layout
    pos = nx.spring_layout(cluster_graph)
    
    # Create Plotly traces for edges
    edge_x = []
    edge_y = []
    for edge in cluster_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Separate edges
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines')

    # Create Plotly traces for nodes
    node_x = []
    node_y = []
    node_text = []
    for node in cluster_graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))  # Use node label as hover text

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='bottom center',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='skyblue',
            size=20,
            line_width=2))

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Cluster Connections Visualization",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))

    return fig  # Return the figure instead of showing it immediately