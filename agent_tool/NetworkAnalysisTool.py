from smolagents import Tool
import os
from astropy import units as u
from pycraf import conversions as cnv
from functools import partial
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.colors import Normalize
import contextily as ctx

# Import all your existing functions
from maintained.src.net_pathloss_calculator import calculate_network_path_loss, visualize_network, save_network_data
from maintained.src.process_network_data import process_network_data
from maintained.src.p2p_path_loss import tx_rx_path_loss

# Set matplotlib parameters for consistent styling
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 13
})

def plot_simple_path_loss_map(link_data, save_path='./figures/pycraf.png'):
    """
    Create a simplified version focusing only on the path loss heatmap
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Get bounds
    all_lons = np.concatenate([link_data['tx_lon'], link_data['rx_lon']])
    all_lats = np.concatenate([link_data['tx_lat'], link_data['rx_lat']])

    lon_margin = (all_lons.max() - all_lons.min()) * 0.05
    lat_margin = (all_lats.max() - all_lats.min()) * 0.05

    lon_min, lon_max = all_lons.min() - lon_margin, all_lons.max() + lon_margin
    lat_min, lat_max = all_lats.min() - lat_margin, all_lats.max() + lat_margin

    # Create transformer
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_inverse = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Transform bounds
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Normalize and plot links
    norm = Normalize(vmin=link_data['path_loss_db'].min(), vmax=link_data['path_loss_db'].max())
    cmap = plt.cm.RdYlBu_r

    for _, link in link_data.iterrows():
        tx_x, tx_y = transformer.transform(link['tx_lon'], link['tx_lat'])
        rx_x, rx_y = transformer.transform(link['rx_lon'], link['rx_lat'])

        color = cmap(norm(link['path_loss_db']))
        ax.plot([tx_x, rx_x], [tx_y, rx_y],
                color=color, linewidth=2, alpha=0.8)

    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap, alpha=0.7)
    except:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Path Loss (dB)', fontsize=12)

    # Fix axis labels
    mid_y = np.mean(ax.get_ylim())
    mid_x = np.mean(ax.get_xlim())

    xticks = ax.get_xticks()
    new_xticks = [transformer_inverse.transform(x, mid_y)[0] for x in xticks]
    ax.set_xticklabels([f"{lon:.2f}" for lon in new_xticks])

    yticks = ax.get_yticks()
    new_yticks = [transformer_inverse.transform(mid_x, y)[1] for y in yticks]
    ax.set_yticklabels([f"{lat:.2f}" for lat in new_yticks])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Network Path Loss Heatmap", fontsize=13)

    plt.savefig(save_path,
                dpi=300, bbox_inches='tight')
    print(f"Path loss map saved to: {save_path}")

    plt.show()
    return fig


class NetworkAnalysisTool(Tool):
    """Tool for network path loss calculation and data processing."""

    name = "NetworkAnalysisTool"
    description = """
    Analyzes network performance by calculating path loss between nodes and processing network data.
    This tool takes file paths to geographic data files and network parameters to calculate path loss, 
    visualize the network, and process the data for optimization.
    """
    inputs = {
        "pop_loc_path": {
            "type": "string",
            "description": "Path to the population location shapefile (.shp)"
        },
        "cn_loc_path": {
            "type": "string",
            "description": "Path to the clustered node location shapefile (.shp)"
        },
        "tbs_loc_path": {
            "type": "string",
            "description": "Path to the TBS location shapefile (.shp)"
        },
        "hap_loc_path": {
            "type": "string",
            "description": "Path to the HAP location shapefile (.shp)"
        },
        "frequency": {
            "type": "number",
            "description": "Frequency value (default: 5)",
            "nullable": True
        },
        "bandwidth": {
            "type": "number",
            "description": "Bandwidth value (default: 10)",
            "nullable": False
        },
        "frequency_unit": {
            "type": "string",
            "description": "Unit for frequency (GHz, MHz, Hz) (default: 'GHz')",
            "nullable": False
        },
        "bandwidth_unit": {
            "type": "string",
            "description": "Unit for bandwidth (GHz, MHz, Hz) (default: 'MHz')",
            "nullable": False
        },
    }
    output_type = "object"

    def forward(self, pop_loc_path: str, cn_loc_path: str, tbs_loc_path: str, hap_loc_path: str,
                frequency: float = 5, bandwidth: float = 10,
                frequency_unit: str = "GHz", bandwidth_unit: str = "MHz",
                ) -> Dict[str, Any]:
        """
        Calculate network path loss and process network data.

        Args:
            pop_loc_path: Path to population location shapefile
            cn_loc_path: Path to clustered node location shapefile
            tbs_loc_path: Path to TBS location shapefile
            hap_loc_path: Path to HAP location shapefile
            frequency: Frequency value
            bandwidth: Bandwidth value
            frequency_unit: Unit for frequency (GHz, MHz, Hz)
            bandwidth_unit: Unit for bandwidth (GHz, MHz, Hz)
        Returns:
            Dictionary containing network analysis results
        """
        # Ensure output directories exist
        os.makedirs('./tmp_data/network', exist_ok=True)

        # Convert frequency and bandwidth to appropriate units
        if frequency_unit == "GHz":
            frequency = frequency * u.GHz
        elif frequency_unit == "MHz":
            frequency = frequency * u.MHz
        else:
            frequency = frequency * u.Hz

        if bandwidth_unit == "GHz":
            bandwidth_Hz = bandwidth * u.GHz
        elif bandwidth_unit == "MHz":
            bandwidth_Hz = bandwidth * u.MHz
        else:
            bandwidth_Hz = bandwidth * u.Hz

        # Create custom path loss function with specified frequency
        custom_path_loss = partial(
            tx_rx_path_loss,
            frequency=frequency,
            G_t=0 * cnv.dBi,
            G_r=10 * cnv.dBi
        )

        # Define output paths
        node_output_path = "./tmp_data/network/nodes.csv"
        link_output_path = "./tmp_data/network/links.csv"
        updated_nodes_output_path = "./tmp_data/network/updated_nodes.csv"

        # Load data from shapefiles
        print("Loading location data from shapefiles...")
        import geopandas as gpd

        pop_loc = gpd.read_file(pop_loc_path)
        cn_loc = gpd.read_file(cn_loc_path)
        tbs_loc = gpd.read_file(tbs_loc_path)
        hap_loc = gpd.read_file(hap_loc_path)

        # Calculate network path loss
        print("Calculating network path loss...")
        node_data, link_data = calculate_network_path_loss(
            pop_loc, cn_loc, tbs_loc, hap_loc, custom_path_loss
        )

        # Visualize network
        visualize_network(node_data, link_data, title='Network Visualization (Incomplete Links)')

        # Save the results
        save_network_data(
            node_data, link_data,
            node_output_path=node_output_path,
            link_output_path=link_output_path
        )

        # Process network data
        print("Processing network data...")
        updated_node_df = process_network_data(
            link_output_path, node_output_path,
            output_path=updated_nodes_output_path,
            demand_threshold=150
        )
        plot_simple_path_loss_map(link_data)

        return {
            # "node_data": node_data,
            # "link_data": link_data,
            # "updated_node_df": updated_node_df,
            # "virgin_node_path": node_output_path,
            "netana_link_path": link_output_path,
            "netana_node_path": updated_nodes_output_path,
            # "bandwidth_Hz": bandwidth_Hz
        }