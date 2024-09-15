import os
import networkx as nx
import torch
from llama_index.core.schema import TextNode
from concurrent.futures import ThreadPoolExecutor
import pickle
from tqdm import tqdm
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

def save_graph(G,save_dir):
    pickle_path = f"{save_dir}/graph.pkl"
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)

def load_graph(save_dir):
    
    with open(f"{save_dir}/graph.pkl", "rb") as f:
        G = pickle.load(f)
    return G

def create_graph(embedded_chunks,
                 chunk_similarities,
                 threshold,
                 save_dir,
                 save
                ):
    
    G = nx.Graph()

    def add_node_to_graph(G, node):
        G.add_node(node.id_, node=node)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(add_node_to_graph, G, chunk) for chunk in embedded_chunks]
        for future in tqdm(futures, total=len(futures),desc=f"adding contexts..."):
            future.result()

    # The following is for a threshold
    indices = torch.triu_indices(chunk_similarities.shape[0], chunk_similarities.shape[1], offset=1,device='cpu')
    mask = chunk_similarities[indices[0], indices[1]] > 0
    filtered_indices = indices[:, mask]
    filtered_weights = chunk_similarities[filtered_indices[0], filtered_indices[1]].tolist()
    filtered_indices_list = [(str(i), str(j), w) for i, j, w in zip(filtered_indices[0].tolist(), filtered_indices[1].tolist(), filtered_weights)]
    G.add_weighted_edges_from(filtered_indices_list)

    if save:
        print(f"saving to disk...")
        save_graph(G,save_dir)

    print(f"Graph created")

    return G

def visualize_graph(G, filename="network_graph.html"):
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
        node_texts.append(f'{node}:# of connections: {len(adj_nodes)} \n{list(dict(adj_nodes).keys())}')  # Add to node_texts list

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

    # Save the figure as an HTML file
    pio.write_html(fig, file=filename, auto_open=True)



def retrieve_embeddings_and_sentences_from_graph(graph, node_ids):
    embeddings = []
    sentences = []
    for node_id in node_ids:
        if node_id in graph.nodes:
            embedding = graph.nodes[node_id]['embedding']
            sentence = graph.nodes[node_id]['sentence']
            embeddings.append(embedding)
            sentences.append(sentence)

    return embeddings,sentences