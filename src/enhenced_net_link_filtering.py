import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u


# Simple Shannon capacity model
def calculate_rate(bandwidth, path_loss_dB, tx_power_watt):
    if hasattr(bandwidth, 'unit'):
        # Handle different units
        if bandwidth.unit == u.GHz:
            bandwidth_Hz = bandwidth.value * 1e9
        elif bandwidth.unit == u.MHz:
            bandwidth_Hz = bandwidth.value * 1e6
        elif bandwidth.unit == u.Hz:
            bandwidth_Hz = bandwidth.value
        else:
            # Try to convert to Hz if the unit system allows
            try:
                bandwidth_Hz = bandwidth.to(u.Hz).value
            except:
                raise ValueError(f"Unsupported bandwidth unit: {bandwidth.unit}. Please use Hz, MHz, or GHz.")
    else:
        # Assume it's already in Hz if no unit is attached
        bandwidth_Hz = bandwidth
    # Shannon capacity formula-based approximation
    noise_floor_dBm = -174 + 10 * math.log10(bandwidth_Hz)  # Thermal noise in dBm
    tx_power_dBm = 10 * math.log10(tx_power_watt * 1000)  # Convert to dBm
    received_power_dBm = tx_power_dBm - path_loss_dB
    snr_dB = received_power_dBm - noise_floor_dBm
    if snr_dB <= 0:
        return 0
    capacity = bandwidth_Hz * math.log2(1 + 10 ** (snr_dB / 10))
    return capacity

def parse_coords(coord):
    """
    Safely parse coordinates whether they're tuples or strings.

    Parameters:
    -----------
    coord : tuple or str
        Coordinates as tuple or string representation

    Returns:
    --------
    tuple or None
        Parsed coordinates as (lon, lat) tuple, or None if parsing fails
    """
    if isinstance(coord, tuple):
        return coord
    elif isinstance(coord, str):
        try:
            return eval(coord)
        except:
            # If eval fails, try to parse as a string representation of a tuple
            try:
                coord = coord.strip('()').split(',')
                return (float(coord[0]), float(coord[1]))
            except:
                return None
    else:
        # For any other unexpected format
        return None


def calculate_refined_direction(lon1, lat1, lon2, lat2, height1=None, height2=None):
    """
    Calculate a more detailed direction classification that includes both
    horizontal and vertical components, resulting in up to 26 possible directions.

    Parameters:
    -----------
    lon1, lat1 : float
        Longitude and latitude of the first point (source/rx)
    lon2, lat2 : float
        Longitude and latitude of the second point (destination/tx)
    height1, height2 : float, optional
        Heights of the points (if provided)

    Returns:
    --------
    direction : str
        One of 26 possible directions including vertical components
    """
    # Calculate horizontal angle
    dlon = lon2 - lon1
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dlon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dlon))
    angle = math.degrees(math.atan2(x, y))
    horizontal_angle = (angle + 360) % 360  # Normalize to 0-360

    # Map to 8 cardinal directions
    h_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    h_index = round(horizontal_angle / 45) % 8
    h_direction = h_directions[h_index]

    # Default to middle (horizontal plane)
    v_direction = 'MID'

    # Check for vertical component if heights are provided
    if height1 is not None and height2 is not None:
        height_diff = height2 - height1

        # Calculate the horizontal distance
        earth_radius = 6371  # km
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        horizontal_distance = earth_radius * math.acos(
            math.sin(lat1_rad) * math.sin(lat2_rad) +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon1_rad - lon2_rad)
        )

        # Calculate the angle of elevation/depression
        if horizontal_distance > 0:
            elevation_angle = math.degrees(math.atan2(height_diff, horizontal_distance * 1000))
        else:
            elevation_angle = 90 if height_diff > 0 else -90

        # Determine vertical direction
        if abs(elevation_angle) < 15:  # Nearly horizontal (+/- 15 degrees)
            v_direction = 'MID'
        elif elevation_angle >= 15 and elevation_angle < 75:  # Upward
            v_direction = 'UP'
        elif elevation_angle >= 75:  # Nearly vertical up
            return 'VERTICAL-UP'  # Special case for nearly vertical
        elif elevation_angle <= -15 and elevation_angle > -75:  # Downward
            v_direction = 'DOWN'
        elif elevation_angle <= -75:  # Nearly vertical down
            return 'VERTICAL-DOWN'  # Special case for nearly vertical

    # Combine horizontal and vertical components
    if v_direction == 'MID':
        return h_direction
    else:
        return f"{h_direction}-{v_direction}"


def filter_links_by_type_and_direction(link_pairs, allowed_link_types, max_links_per_direction=3,
                                       tx_power_dict=None):
    """
    Filter links to keep only the allowed link types and the best N (lowest path loss) in each direction,
    considering both rx_id and tx_id perspectives to ensure comprehensive coverage.

    Parameters:
    -----------
    link_pairs : list of dict
        List of link information dictionaries
    allowed_link_types : list
        List of allowed link types (e.g., ['hap-tbs', 'tbs-tbs', 'pop-tbs'])
    max_links_per_direction : int, default=3
        Maximum number of links to keep per direction
    tx_power_dict : dict, default=None
        Dictionary mapping node types to their transmission power in watts

    Returns:
    --------
    best_links : list of dict
        Filtered list of best links per direction with redundancies removed
    """
    if tx_power_dict is None:
        # Default power values if not provided
        tx_power_dict = {
            'hap': 20.0,  # Higher power for HAPs
            'tbs': 5.0,  # Lower power for TBSs
            'pop': 5.0  # Same as TBS by default
        }

    # Create dictionaries to store links for both perspectives
    rx_directional_links = defaultdict(lambda: defaultdict(list))
    tx_directional_links = defaultdict(lambda: defaultdict(list))

    # Filter for allowed link types
    filtered_links = [link for link in link_pairs if link.get('link_type', '') in allowed_link_types]

    # Process each link from both perspectives
    for link in filtered_links:
        try:
            # Extract coordinates and heights
            rx_coords = parse_coords(link.get('rx_coords', None))
            tx_coords = parse_coords(link.get('tx_coords', None))

            if not rx_coords or not tx_coords:
                continue  # Skip if coordinates are not valid

            rx_lon, rx_lat = rx_coords
            tx_lon, tx_lat = tx_coords

            # Get heights if available
            rx_height = link.get('rx_height', None)
            tx_height = link.get('tx_height', None)

            # Get node IDs
            rx_id = link.get('rx_id', link.get('rx_index', None))
            tx_id = link.get('tx_id', link.get('tx_index', None))

            if rx_id is None or tx_id is None:
                continue  # Skip if IDs are not valid

            # 1. Process from RX perspective (incoming links to rx_id)
            rx_direction = calculate_refined_direction(
                rx_lon, rx_lat, tx_lon, tx_lat, rx_height, tx_height
            )
            rx_directional_links[rx_id][rx_direction].append(link)

            # 2. Process from TX perspective (outgoing links from tx_id)
            # For the TX perspective, we need to reverse the direction calculation
            tx_direction = calculate_refined_direction(
                tx_lon, tx_lat, rx_lon, rx_lat, tx_height, rx_height
            )

            # Create a reversed link (same link but viewed from tx perspective)
            reverse_link = link.copy()
            # Add a marker to indicate this is reversed (for merging later)
            reverse_link['is_reversed'] = True
            tx_directional_links[tx_id][tx_direction].append(reverse_link)

        except (KeyError, ValueError, TypeError) as e:
            continue

    # Create sets to track selected links (to avoid redundancy)
    selected_link_pairs = set()  # (rx_id, tx_id) pairs that have been selected
    best_links = []

    # Helper function to create a link entry
    def create_link_entry(link, direction, is_reversed=False):
        # If reversed, swap rx and tx
        if is_reversed:
            rx_id = link.get('tx_id', link.get('tx_index', None))
            tx_id = link.get('rx_id', link.get('rx_index', None))
            rx_type = link.get('tx_type', 'unknown')
            tx_type = link.get('rx_type', 'unknown')
            rx_lon = link.get('tx_lon', None)
            rx_lat = link.get('tx_lat', None)
            tx_lon = link.get('rx_lon', None)
            tx_lat = link.get('rx_lat', None)
            rx_coords = parse_coords(link.get('tx_coords'))
            tx_coords = parse_coords(link.get('rx_coords'))
            rx_height = link.get('tx_height', None)
            tx_height = link.get('rx_height', None)
        else:
            rx_id = link.get('rx_id', link.get('rx_index', None))
            tx_id = link.get('tx_id', link.get('tx_index', None))
            rx_type = link.get('rx_type', 'unknown')
            tx_type = link.get('tx_type', 'unknown')
            rx_lon = link.get('rx_lon', None)
            rx_lat = link.get('rx_lat', None)
            tx_lon = link.get('tx_lon', None)
            tx_lat = link.get('tx_lat', None)
            rx_coords = parse_coords(link.get('rx_coords'))
            tx_coords = parse_coords(link.get('tx_coords'))
            rx_height = link.get('rx_height', None)
            tx_height = link.get('tx_height', None)

        # Create link entry with standardized format
        link_entry = {
            'rx_id': rx_id,
            'tx_id': tx_id,
            'tx_type': tx_type,
            'rx_type': rx_type,
            'tx_lon': tx_lon,
            'tx_lat': tx_lat,
            'rx_lon': rx_lon,
            'rx_lat': rx_lat,
            'rx_coords': rx_coords,
            'tx_coords': tx_coords,
            'tx_height': tx_height,
            'rx_height': rx_height,
            'link_type': link.get('link_type', f"{tx_type}-{rx_type}"),
            'path_loss_db': float(link.get('path_loss_db', link.get('path_loss_dB', float('inf')))),
            'direction': direction
        }

        # Add distance if available
        if 'distance_km' in link:
            link_entry['distance_km'] = link['distance_km']

        return link_entry

    # Function to process links from a perspective
    def process_links(directional_links, is_reversed=False):
        processed_links = []

        for node_id, directions in directional_links.items():
            for direction, links in directions.items():
                # Sort links by path loss (ascending)
                sorted_links = sorted(links,
                                      key=lambda x: float(x.get('path_loss_db', x.get('path_loss_dB', float('inf')))))

                # Take up to max_links_per_direction
                num_links = min(max_links_per_direction, len(sorted_links))

                # Add the best N links
                for i in range(num_links):
                    if i >= len(sorted_links):
                        break  # Safety check

                    link = sorted_links[i]

                    # Determine rx_id and tx_id (possibly reversed)
                    if is_reversed:
                        rx_id = link.get('tx_id', link.get('tx_index', None))
                        tx_id = link.get('rx_id', link.get('rx_index', None))
                    else:
                        rx_id = link.get('rx_id', link.get('rx_index', None))
                        tx_id = link.get('tx_id', link.get('tx_index', None))

                    # Check if this link pair has already been selected
                    link_pair = (min(rx_id, tx_id), max(rx_id, tx_id))
                    if link_pair not in selected_link_pairs:
                        selected_link_pairs.add(link_pair)
                        entry = create_link_entry(link, direction, is_reversed)
                        processed_links.append(entry)

        return processed_links

    # Process links from both perspectives
    rx_links = process_links(rx_directional_links, is_reversed=False)
    tx_links = process_links(tx_directional_links, is_reversed=True)

    # Combine all links
    best_links = rx_links + tx_links

    return best_links


def calculate_link_capacity(links, bandwidth_Hz=10e6, tx_power_dict=None):
    """
    Calculate transmission capacity for each link with node-type specific power.

    Parameters:
    -----------
    links : list of dict
        List of link information dictionaries
    tx_power_dict : dict, default=None
        Dictionary mapping node types to their transmission power in watts

    Returns:
    --------
    links_with_capacity : list of dict
        Links with added capacity information
    """
    if tx_power_dict is None:
        # Default power values if not provided
        tx_power_dict = {
            'hap': 20.0,  # Higher power for HAPs
            'tbs': 5.0,  # Lower power for TBSs
            'pop': 5.0  # Same as TBS by default
        }


    links_with_capacity = []
    for link in links:
        # Get path loss (handle different key names)
        path_loss_dB = float(link.get('path_loss_db', link.get('path_loss_dB', 0)))

        # Get transmitter type and determine power
        tx_type = link.get('tx_type', '').lower()

        # Default to tbs power if type not recognized
        tx_power_watt = tx_power_dict.get(tx_type, tx_power_dict.get('tbs', 5.0))

        # Calculate capacity with 10 MHz bandwidth
        bandwidth_Hz = bandwidth_Hz
        trans_cap_bps = calculate_rate(bandwidth_Hz, path_loss_dB, tx_power_watt)

        # Convert to Mbps for readability
        cap_mbps = trans_cap_bps / 1e6

        # Create a new link entry with capacity information
        new_link = link.copy()
        new_link['cap_mbps'] = cap_mbps
        new_link['tx_power_watt'] = tx_power_watt

        links_with_capacity.append(new_link)

    return links_with_capacity


def add_node_demands(filtered_links, node_data):
    """
    Add demand information from node data to the links data.

    Parameters:
    -----------
    filtered_links : list of dict
        Filtered links with best connections
    node_data : DataFrame
        Information about all nodes with demand data

    Returns:
    --------
    links_with_demand : list of dict
        Links with added node demand information
    """
    # Create a lookup dictionary for node demands
    node_demands = {}
    for _, node in node_data.iterrows():
        node_id = node.get('node_id')
        if node_id is not None:
            node_demands[node_id] = node.get('demand', 0.0)

    # Add demand information to links
    links_with_demand = []
    for link in filtered_links:
        new_link = link.copy()

        # Add demand information for both rx and tx nodes
        rx_id = link.get('rx_id')
        tx_id = link.get('tx_id')

        new_link['rx_demand'] = node_demands.get(rx_id, 0.0)
        new_link['tx_demand'] = node_demands.get(tx_id, 0.0)

        links_with_demand.append(new_link)

    return links_with_demand


def visualize_network(nodes_df, filtered_links_df, save_path="vis_links_png"):
    """
    Visualize the filtered network in 3D.

    Parameters:
    -----------
    nodes_df : DataFrame
        Information about all nodes
    filtered_links_df : DataFrame
        Filtered links with best connections
    save_path : str, optional
        Path to save the visualization
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a node dictionary for quick lookup
    node_dict = {}
    for _, node in nodes_df.iterrows():
        node_id = node.get('node_id')
        if node_id is not None:
            node_dict[node_id] = {
                'lon': node.get('lon', 0),
                'lat': node.get('lat', 0),
                'height': node.get('height', 0),
                'type': node.get('node_type', 'unknown')
            }

    # Define colors for different node types
    node_colors = {
        'hap': 'red',
        'tbs': 'blue',
        'pop': 'green'
    }

    # Plot nodes
    node_xs, node_ys, node_zs = [], [], []
    node_colors_list = []

    for node_id, node_info in node_dict.items():
        node_xs.append(node_info['lon'])
        node_ys.append(node_info['lat'])
        node_zs.append(node_info['height'])
        node_colors_list.append(node_colors.get(node_info['type'].lower(), 'gray'))

    ax.scatter(node_xs, node_ys, node_zs, c=node_colors_list, s=50, alpha=0.8)

    # Plot links
    for _, link in filtered_links_df.iterrows():
        tx_id = link.get('tx_id')
        rx_id = link.get('rx_id')

        if tx_id in node_dict and rx_id in node_dict:
            tx_node = node_dict[tx_id]
            rx_node = node_dict[rx_id]

            # Get coordinates
            x = [tx_node['lon'], rx_node['lon']]
            y = [tx_node['lat'], rx_node['lat']]
            z = [tx_node['height'], rx_node['height']]

            # Set link color based on link_type
            link_type = link.get('link_type', '')
            if 'hap-tbs' in link_type:
                color = 'red'
            elif 'tbs-tbs' in link_type:
                color = 'blue'
            elif 'pop-tbs' in link_type:
                color = 'green'
            else:
                color = 'gray'

            # Plot the link
            ax.plot(x, y, z, color=color, linewidth=1, alpha=0.6)

    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height')
    ax.set_title('3D Network Visualization')

    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='red', label='HAP'),
        Patch(facecolor='blue', label='TBS'),
        Patch(facecolor='green', label='POP'),
        Line2D([0], [0], color='red', lw=2, label='HAP-TBS Link'),
        Line2D([0], [0], color='blue', lw=2, label='TBS-TBS Link'),
        Line2D([0], [0], color='green', lw=2, label='POP-TBS Link')
    ]

    ax.legend(handles=legend_elements, loc='upper right')

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    return fig


def create_network_graph(filtered_links_df):
    """
    Create a NetworkX graph from the filtered links.

    Parameters:
    -----------
    filtered_links_df : DataFrame
        Filtered links with best connections

    Returns:
    --------
    G : NetworkX Graph
        Graph representation of the network
    """
    # Create an empty graph
    G = nx.DiGraph()

    # Add edges with capacity information
    for _, link in filtered_links_df.iterrows():
        tx_id = link.get('tx_id')
        rx_id = link.get('rx_id')

        if tx_id is not None and rx_id is not None:
            # Get link attributes
            capacity = link.get('cap_mbps', 0)
            link_type = link.get('link_type', '')

            # Add the edge with attributes
            G.add_edge(
                tx_id,
                rx_id,
                capacity=capacity,
                link_type=link_type
            )

    return G


def filter_hap_links(links, max_links_per_hap=10):
    """
    Further filter HAP-TBS links to limit the number of connections per HAP.

    Parameters:
    -----------
    links : list of dict
        List of link information dictionaries
    max_links_per_hap : int, default=10
        Maximum number of outgoing links per HAP

    Returns:
    --------
    filtered_links : list of dict
        Links with HAP connections limited
    """
    # First separate HAP-TBS links from other links
    hap_tbs_links = [link for link in links if link.get('link_type') == 'hap-tbs']
    other_links = [link for link in links if link.get('link_type') != 'hap-tbs']

    # Group HAP-TBS links by HAP ID
    hap_links = defaultdict(list)
    for link in hap_tbs_links:
        tx_id = link.get('tx_id')
        if link.get('tx_type', '').lower() == 'hap':
            hap_links[tx_id].append(link)

    # Filter to keep only the best links per HAP
    filtered_hap_links = []
    for hap_id, links in hap_links.items():
        # Sort by capacity (descending)
        sorted_links = sorted(links, key=lambda x: x.get('cap_mbps', 0), reverse=True)
        # Keep only the top links
        filtered_hap_links.extend(sorted_links[:max_links_per_hap])

    # Combine with other links
    filtered_links = other_links + filtered_hap_links

    return filtered_links


def filter_by_coverage_area(links, max_distance_km=10):
    """
    Filter HAP-TBS links based on a maximum distance/coverage area.

    Parameters:
    -----------
    links : list of dict
        List of link information dictionaries
    max_distance_km : float, default=10
        Maximum distance in kilometers for HAP-TBS links

    Returns:
    --------
    filtered_links : list of dict
        Links filtered by distance
    """
    filtered_links = []

    # First separate HAP-TBS links from other links
    hap_tbs_links = [link for link in links if link.get('link_type') == 'hap-tbs']
    other_links = [link for link in links if link.get('link_type') != 'hap-tbs']

    for link in hap_tbs_links:
        # Calculate distance if not already present
        if 'distance_km' not in link:
            rx_coords = parse_coords(link.get('rx_coords'))
            tx_coords = parse_coords(link.get('tx_coords'))

            if rx_coords and tx_coords:
                rx_lon, rx_lat = rx_coords
                tx_lon, tx_lat = tx_coords

                # Calculate great-circle distance
                earth_radius = 6371  # km
                lat1_rad = math.radians(tx_lat)
                lat2_rad = math.radians(rx_lat)
                lon1_rad = math.radians(tx_lon)
                lon2_rad = math.radians(rx_lon)

                distance = earth_radius * math.acos(
                    math.sin(lat1_rad) * math.sin(lat2_rad) +
                    math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon1_rad - lon2_rad)
                )

                link['distance_km'] = distance

        # For HAP-TBS links, check distance
        if link.get('distance_km', float('inf')) <= max_distance_km:
            filtered_links.append(link)

    # Combine with other links
    filtered_links.extend(other_links)

    return filtered_links


def filter_by_grid_coverage(links, node_data, grid_size=5):
    """
    Filter HAP-TBS links to ensure coverage across a grid.

    Parameters:
    -----------
    links : list of dict
        List of link information dictionaries
    node_data : DataFrame
        Information about all nodes
    grid_size : float, default=5
        Size of grid cells in coordinate units

    Returns:
    --------
    filtered_links : list of dict
        Links filtered to ensure grid coverage
    """
    # Separate HAP-TBS links from other links
    hap_tbs_links = [link for link in links if link.get('link_type') == 'hap-tbs']
    other_links = [link for link in links if link.get('link_type') != 'hap-tbs']

    # Create a grid structure
    grid_cells = defaultdict(list)

    # Assign TBS nodes to grid cells
    tbs_nodes = node_data[node_data['node_type'].str.lower() == 'tbs']
    for _, node in tbs_nodes.iterrows():
        lon = node['lon']
        lat = node['lat']
        cell_x = int(lon / grid_size)
        cell_y = int(lat / grid_size)
        grid_cells[(cell_x, cell_y)].append(node['node_id'])

    # For each HAP, ensure coverage across grid cells
    hap_ids = set(link['tx_id'] for link in hap_tbs_links
                  if link.get('tx_type', '').lower() == 'hap')

    filtered_hap_links = []
    covered_cells = set()

    # First assign HAPs to their best cell coverage
    for hap_id in hap_ids:
        hap_links = [link for link in hap_tbs_links
                     if link['tx_id'] == hap_id]

        # Track which cells this HAP can cover
        hap_coverage = defaultdict(list)

        for link in hap_links:
            rx_id = link['rx_id']
            # Find which cell this TBS is in
            for cell, tbs_ids in grid_cells.items():
                if rx_id in tbs_ids:
                    hap_coverage[cell].append(link)
                    break

        # For each cell this HAP can cover, keep the best link
        for cell, cell_links in hap_coverage.items():
            if cell not in covered_cells:
                # Sort by capacity
                sorted_links = sorted(cell_links,
                                      key=lambda x: x.get('cap_mbps', 0),
                                      reverse=True)
                if sorted_links:
                    filtered_hap_links.append(sorted_links[0])
                    covered_cells.add(cell)

    # Now add additional links for uncovered cells
    for cell, tbs_ids in grid_cells.items():
        if cell not in covered_cells and tbs_ids:
            # Find all HAP-TBS links to this cell
            cell_links = []
            for link in hap_tbs_links:
                if link['rx_id'] in tbs_ids:
                    cell_links.append(link)

            # Sort by capacity
            sorted_links = sorted(cell_links,
                                  key=lambda x: x.get('cap_mbps', 0),
                                  reverse=True)
            if sorted_links:
                filtered_hap_links.append(sorted_links[0])
                covered_cells.add(cell)

    # Combine with other links
    filtered_links = other_links + filtered_hap_links

    return filtered_links


def filter_network_links(node_data, link_data,
                         allowed_link_types=['hap-tbs', 'tbs-tbs', 'pop-tbs'],
                         max_links_per_direction=3,
                         min_capacity_mbps=1.0,
                         tx_power_dict=None,
                         max_links_per_hap=10,
                         max_hap_distance_km=15,
                         apply_grid_filtering=True,
                         grid_size=5,
                         bandwidth_Hz=10e6,
                         visualize=True):
    """
    Filter network links by selecting the best links per direction and capacity.
    Implements a type-specific power allocation and directional filtering.
    """

    # Convert DataFrame to list of dicts for processing
    link_pairs = link_data.to_dict('records')

    # Check and set up transmitter power dictionary
    if tx_power_dict is None:
        # Default power values
        tx_power_dict = {
            'hap': 20.0,  # Higher power for HAPs
            'tbs': 5.0,  # Lower power for TBSs
            'pop': 5.0  # Same as TBS by default
        }

        # Update power for POP nodes if height matches TBS
        pop_nodes = node_data[node_data['node_type'].str.lower() == 'pop']
        tbs_nodes = node_data[node_data['node_type'].str.lower() == 'tbs']

        # If any POP has the same height as TBS, use TBS power for that POP
        if not pop_nodes.empty and not tbs_nodes.empty:
            tbs_height = tbs_nodes['height'].values[0] if len(tbs_nodes) > 0 else None
            if tbs_height is not None:
                for _, pop_node in pop_nodes.iterrows():
                    if pop_node['height'] == tbs_height:
                        node_id = pop_node['node_id']

    # Filter best N links per direction, considering both perspectives
    best_directional_links = filter_links_by_type_and_direction(
        link_pairs,
        allowed_link_types,
        max_links_per_direction,
        tx_power_dict
    )

    # Calculate capacity for each link
    links_with_capacity = calculate_link_capacity(best_directional_links, bandwidth_Hz, tx_power_dict)

    # Filter by minimum capacity if specified
    if min_capacity_mbps > 0:
        filtered_links = [link for link in links_with_capacity if link.get('cap_mbps', 0) >= min_capacity_mbps]
    else:
        filtered_links = links_with_capacity

    # Apply HAP-specific filtering to reduce HAP-TBS links
    filtered_links = filter_hap_links(filtered_links, max_links_per_hap)

    # Apply distance-based filtering for HAP-TBS links
    filtered_links = filter_by_coverage_area(filtered_links, max_hap_distance_km)

    # Apply grid-based coverage filtering if requested
    if apply_grid_filtering:
        filtered_links = filter_by_grid_coverage(filtered_links, node_data, grid_size)

    filtered_links_before = len(filtered_links)
    filtered_links = [link for link in filtered_links if not
    (link.get('tx_type', '').lower() == 'tbs' and
     link.get('rx_type', '').lower() == 'pop')]

    # Add node demand information
    links_with_demand = add_node_demands(filtered_links, node_data)

    # Convert to DataFrame
    filtered_links_df = pd.DataFrame(links_with_demand)


    return filtered_links_df