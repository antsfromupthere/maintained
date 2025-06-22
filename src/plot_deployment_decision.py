import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Set matplotlib parameters for consistent styling
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 13
})


def visualize_connectivity_3d(nodes_csv_path="./tmp_data/network/nodes.csv",
                              deployed_nodes_csv_path="./optimization_results/deployed_nodes.csv",
                              active_links_csv_path="./optimization_results/active_links.csv",
                              links_csv_path="./alpha_data/network/links.csv"):
    """
    Create a clean 3D visualization showing network connectivity with all CN connection links.

    Parameters:
    -----------
    nodes_csv_path : str
        Path to nodes.csv file
    deployed_nodes_csv_path : str
        Path to deployed_nodes.csv file
    active_links_csv_path : str
        Path to active_links.csv file
    links_csv_path : str
        Path to links.csv file

    Returns:
    --------
    fig : matplotlib Figure
        The 3D visualization figure
    """
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 13
    })
    # Load data
    df_nodes = pd.read_csv(nodes_csv_path)
    deployed_nodes_df = pd.read_csv(deployed_nodes_csv_path)
    active_links_df = pd.read_csv(active_links_csv_path)
    links_df = pd.read_csv(links_csv_path)

    # Get deployed node IDs
    deployed_node_ids = set(deployed_nodes_df['node_id'].tolist())

    # Find CN connections to deployed infrastructure
    cn_nodes = df_nodes[df_nodes['node_type'] == 'cn']
    cn_connections = {}

    for _, cn_node in cn_nodes.iterrows():
        cn_id = cn_node['node_id']
        # Find links where CN is receiver and transmitter is deployed
        cn_rx_links = links_df[links_df['rx_id'] == cn_id]
        deployed_tx_links = cn_rx_links[cn_rx_links['tx_id'].isin(deployed_node_ids)]

        if not deployed_tx_links.empty:
            # Get best connection (minimum path loss)
            best_link = deployed_tx_links.sort_values(by='path_loss_db').iloc[0]
            cn_connections[cn_id] = {
                'connected_to': best_link['tx_id'],
                'infrastructure_type': df_nodes[df_nodes['node_id'] == best_link['tx_id']]['node_type'].iloc[0]
            }

    # Create 3D figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot deployed infrastructure nodes
    legend_added = set()

    for _, node in deployed_nodes_df.iterrows():
        node_type = node['node_type'].lower()

        if node_type == 'hap':
            label = 'HAP' if 'HAP' not in legend_added else ""
            ax.scatter(node['lon'], node['lat'], node['height'],
                       c='red', marker='^', s=120, alpha=0.9,
                       edgecolors='black', linewidth=0.5, label=label)
            legend_added.add('HAP')

        elif node_type == 'tbs':
            label = 'TBS' if 'TBS' not in legend_added else ""
            ax.scatter(node['lon'], node['lat'], node['height'],
                       c='blue', marker='o', s=80, alpha=0.9,
                       edgecolors='black', linewidth=0.5, label=label)
            legend_added.add('TBS')

        elif node_type == 'pop':
            label = 'Existing TBS' if 'Existing TBS' not in legend_added else ""
            ax.scatter(node['lon'], node['lat'], node['height'],
                       c='green', marker='s', s=100, alpha=0.9,
                       edgecolors='black', linewidth=0.5, label=label)
            legend_added.add('Existing TBS')

    # Plot customer nodes
    for _, cn_node in cn_nodes.iterrows():
        cn_id = cn_node['node_id']

        if cn_id in cn_connections:
            # Connected CN
            label = 'User' if 'User' not in legend_added else ""
            ax.scatter(cn_node['lon'], cn_node['lat'], cn_node['height'],
                       c='lightgreen', marker='D', s=50, alpha=0.8,
                       edgecolors='darkgreen', linewidth=1, label=label)
            legend_added.add('User')
        else:
            # Unconnected CN
            label = 'Unconnected CN' if 'Unconnected CN' not in legend_added else ""
            ax.scatter(cn_node['lon'], cn_node['lat'], cn_node['height'],
                       c='red', marker='D', s=50, alpha=0.8,
                       edgecolors='darkred', linewidth=1, label=label)
            legend_added.add('Unconnected CN')

    # Plot backbone links (infrastructure to infrastructure)
    for _, link in active_links_df.iterrows():
        tx_node = df_nodes[df_nodes['node_id'] == link['tx_id']].iloc[0]
        rx_node = df_nodes[df_nodes['node_id'] == link['rx_id']].iloc[0]

        ax.plot([tx_node['lon'], rx_node['lon']],
                [tx_node['lat'], rx_node['lat']],
                [tx_node['height'], rx_node['height']],
                color='darkblue', linewidth=2.5, alpha=0.8)

    # Add backbone link to legend
    if active_links_df is not None and len(active_links_df) > 0:
        ax.plot([], [], [], color='darkblue', linewidth=2.5, alpha=0.8)

    # Plot CN access links
    link_colors = {'hap': 'red', 'tbs': 'blue', 'pop': 'green'}

    for cn_id, connection in cn_connections.items():
        cn_node = df_nodes[df_nodes['node_id'] == cn_id].iloc[0]
        infra_node = df_nodes[df_nodes['node_id'] == connection['connected_to']].iloc[0]

        # Get color based on infrastructure type
        infra_type = connection['infrastructure_type'].lower()
        link_color = link_colors.get(infra_type, 'gray')

        # Plot access link
        ax.plot([cn_node['lon'], infra_node['lon']],
                [cn_node['lat'], infra_node['lat']],
                [cn_node['height'], infra_node['height']],
                color=link_color, linewidth=1.5, alpha=0.6, linestyle='--')

    # Add access links to legend
    if cn_connections:
        ax.plot([], [], [], color='gray', linewidth=1.5, alpha=0.6,
                linestyle='--')

    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height (m)')
    ax.set_title('Network Deployment Visualization')

    # Invert x-axis so longitude increases away from intersection point
    ax.invert_xaxis()

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    # Clean up 3D appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    plt.subplots_adjust(
        left=0.07,  # Left margin (0.0 to 1.0)
        bottom=0.01,  # Bottom margin
        right=0.99,  # Right margin
        top=0.99,  # Top margin
        wspace=0,  # Width spacing between subplots (if multiple)
        hspace=0  # Height spacing between subplots (if multiple)
    )
    save_path = "./figures/coverage.png"
    plt.savefig(save_path, dpi=300)
    print(f"Path loss map saved to: {save_path}")


    plt.show()

    return fig

