import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
from astropy import units as u


def calculate_network_path_loss(pop_loc, cn_loc, tbs_loc, hap_loc, tx_rx_path_loss):
    """
    Calculate path loss between different types of network nodes.

    Parameters:
    -----------
    pop_loc : GeoDataFrame
        Population tower locations
    cn_loc : GeoDataFrame
        Customer node locations
    tbs_loc : GeoDataFrame
        Terrestrial base station locations
    hap_loc : GeoDataFrame
        High altitude platform locations
    tx_rx_path_loss : function
        Function to calculate path loss between two points

    Returns:
    --------
    node_data : DataFrame
        Information about all nodes with unified indexing
    link_data : DataFrame
        Information about all possible links between nodes
    """
    # Define heights for different node types (in meters)
    heights = {
        'pop': 50.0,  # Population tower height
        'cn': 1.5,  # Customer node height (ground level)
        'tbs': 50.0,  # Terrestrial base station height
        'hap': 10000.0  # High altitude platform height (20km)
    }

    # Step 1: Create a unified node index for all types of nodes
    # First, add type information to each GeoDataFrame
    pop_loc['node_type'] = 'pop'
    cn_loc['node_type'] = 'cn'
    tbs_loc['node_type'] = 'tbs'
    hap_loc['node_type'] = 'hap'

    # Combine all nodes into a single GeoDataFrame
    all_nodes = pd.concat([pop_loc, cn_loc, tbs_loc, hap_loc], ignore_index=True)

    # Create a clean node_data DataFrame with uniform columns
    node_data = pd.DataFrame({
        'node_id': range(len(all_nodes)),
        'node_type': all_nodes['node_type'],
        'lon': all_nodes.geometry.x,
        'lat': all_nodes.geometry.y,
        'height': all_nodes['node_type'].map(heights),
        'demand': all_nodes['demand']
    })

    # Create a dictionary to map coordinates to node_id
    # Use (lon, lat) as key and node_id as value
    coord_to_id = {(row['lon'], row['lat']): row['node_id'] for _, row in node_data.iterrows()}

    # Step 2: Create all valid combinations of links
    # valid_links = [
    #     ('pop', 'hap'), ('pop', 'cn'), ('pop', 'tbs'),
    #     ('tbs', 'tbs'), ('tbs', 'hap'), ('tbs', 'cn'),
    #     ('hap', 'tbs'), ('hap', 'hap'), ('hap', 'cn')
    # ]

    valid_links = [
        ('pop', 'hap'), ('pop', 'cn'), ('pop', 'tbs'),
        ('tbs', 'tbs'), ('tbs', 'hap'), ('tbs', 'cn'),
        ('hap', 'tbs'), ('hap', 'cn')
    ]

    # Initialize an empty list to store link data
    links = []

    # For each valid link type, create all possible connections
    for tx_type, rx_type in valid_links:
        tx_nodes = node_data[node_data['node_type'] == tx_type]
        rx_nodes = node_data[node_data['node_type'] == rx_type]

        # If link type is from a node type to itself (e.g., tbs-tbs, hap-hap),
        # make sure we don't create self-loops
        if tx_type == rx_type:
            for i, tx in tx_nodes.iterrows():
                for j, rx in rx_nodes.iterrows():
                    if tx['node_id'] != rx['node_id']:  # Skip self-loops
                        # Calculate path loss
                        path_loss, _ = tx_rx_path_loss(
                            tx['lon'], tx['lat'],
                            rx['lon'], rx['lat'],
                            tx['height'] * u.m, rx['height'] * u.m
                        )

                        # Store link information
                        links.append({
                            'tx_id': tx['node_id'],
                            'rx_id': rx['node_id'],
                            'tx_type': tx_type,
                            'rx_type': rx_type,
                            'tx_lon': tx['lon'],
                            'tx_lat': tx['lat'],
                            'rx_lon': rx['lon'],
                            'rx_lat': rx['lat'],
                            'rx_coords': (rx['lon'], rx['lat']),
                            'tx_coords': (tx['lon'], tx['lat']),
                            'tx_height': tx['height'],
                            'rx_height': rx['height'],
                            'link_type': f"{tx_type}-{rx_type}",
                            'path_loss_db': path_loss
                        })
        else:
            # For different node types, create all links
            for _, tx in tx_nodes.iterrows():
                for _, rx in rx_nodes.iterrows():
                    # Calculate path loss
                    path_loss, _ = tx_rx_path_loss(
                        tx['lon'], tx['lat'],
                        rx['lon'], rx['lat'],
                        tx['height'] * u.m, rx['height'] * u.m
                    )

                    # Store link information
                    links.append({
                        'tx_id': tx['node_id'],
                        'rx_id': rx['node_id'],
                        'tx_type': tx_type,
                        'rx_type': rx_type,
                        'tx_lon': tx['lon'],
                        'tx_lat': tx['lat'],
                        'rx_lon': rx['lon'],
                        'rx_lat': rx['lat'],
                        'rx_coords': (rx['lon'], rx['lat']),
                        'tx_coords': (tx['lon'], tx['lat']),
                        'tx_height': tx['height'],
                        'rx_height': rx['height'],
                        'link_type': f"{tx_type}-{rx_type}",
                        'path_loss_db': path_loss
                    })

    # Convert links list to DataFrame
    link_data = pd.DataFrame(links)

    return node_data, link_data


def visualize_network(node_data, link_data=None, title='Network Visualization', max_links_to_plot=1000, sample_fraction=0.05):
    """
    Visualize the network nodes and optionally some links.

    Parameters:
    -----------
    node_data : DataFrame
        Information about all nodes
    link_data : DataFrame, optional
        Information about links between nodes
    max_links_to_plot : int, default=1000
        Maximum number of links to plot (to avoid overcrowding)
    sample_fraction : float, default=0.05
        Fraction of links to sample if there are too many
    """
    # Create a GeoDataFrame for nodes
    geometry = [Point(lon, lat) for lon, lat in zip(node_data['lon'], node_data['lat'])]
    gdf_nodes = gpd.GeoDataFrame(node_data, geometry=geometry, crs="EPSG:4326")

    # Convert to Web Mercator for visualization
    gdf_nodes_3857 = gdf_nodes.to_crs(epsg=3857)

    # Create color mapping for node types
    color_map = {
        'pop': 'red',
        'cn': 'blue',
        'tbs': 'green',
        'hap': 'purple'
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot nodes by type
    for node_type, color in color_map.items():
        subset = gdf_nodes_3857[gdf_nodes_3857['node_type'] == node_type]
        subset.plot(ax=ax, color=color, label=node_type.upper(), markersize=30 if node_type == 'hap' else 10)

    # Plot a sample of links if provided
    if link_data is not None and not link_data.empty:
        if len(link_data) > max_links_to_plot:
            # Sample a fraction of links to avoid overcrowding the plot
            sample_size = min(max_links_to_plot, int(len(link_data) * sample_fraction))
            link_sample = link_data.sample(sample_size, random_state=42)
        else:
            link_sample = link_data

        # Convert node coordinates to Web Mercator
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Plot each link in the sample
        for _, link in link_sample.iterrows():
            # Transform coordinates
            tx_x, tx_y = transformer.transform(link['tx_lon'], link['tx_lat'])
            rx_x, rx_y = transformer.transform(link['rx_lon'], link['rx_lat'])

            # Plot the link with color based on link type
            link_type = link['link_type']
            if 'hap' in link_type:
                color = 'purple'
                alpha = 0.3
            elif 'tbs' in link_type:
                color = 'green'
                alpha = 0.3
            elif 'pop' in link_type:
                color = 'red'
                alpha = 0.3
            else:
                color = 'gray'
                alpha = 0.2

            ax.plot([tx_x, rx_x], [tx_y, rx_y], color=color, alpha=alpha, linewidth=0.5)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    # Add legend
    ax.legend(title="Node Types")

    # Set title
    ax.set_title(title)

    # Transform tick labels from Web Mercator to longitude/latitude
    transformer_inverse = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Get the current tick positions
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Calculate the center of the plot for more accurate transformation
    mid_y = np.mean(ax.get_ylim())
    mid_x = np.mean(ax.get_xlim())

    # Transform x-axis ticks to longitude
    lon_labels = []
    for x in xticks:
        lon, _ = transformer_inverse.transform(x, mid_y)
        lon_labels.append(f"{lon:.2f}")

    # Transform y-axis ticks to latitude
    lat_labels = []
    for y in yticks:
        _, lat = transformer_inverse.transform(mid_x, y)
        lat_labels.append(f"{lat:.2f}")

    # Set the new tick labels
    ax.set_xticklabels(lon_labels)
    ax.set_yticklabels(lat_labels)

    # Add axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()

    return fig


def save_network_data(node_data, link_data, node_output_path, link_output_path):
    """
    Save node and link data to CSV files.

    Parameters:
    -----------
    node_data : DataFrame
        Information about all nodes
    link_data : DataFrame
        Information about links between nodes
    node_output_path : str
        Path to save node data
    link_output_path : str
        Path to save link data
    """
    # Save node data
    node_data.to_csv(node_output_path, index=False)

    # Save link data
    link_data.to_csv(link_output_path, index=False)

    print(f"Node data saved to: {node_output_path}")
    print(f"Link data saved to: {link_output_path}")
