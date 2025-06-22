import pandas as pd
import numpy as np


def process_network_data(links_path, nodes_path, output_path='updated_nodes.csv', demand_threshold=None):
    """
    Process network data to update the demand of transmitter nodes based on the
    connected 'cn' nodes with minimum path loss, with load balancing based on a demand threshold.

    Parameters:
    links_path (str): Path to the links CSV file
    nodes_path (str): Path to the nodes CSV file
    output_path (str): Path to save the updated nodes CSV file, default is 'updated_nodes.csv'
    demand_threshold (float): Maximum demand threshold for a tx node. If None, no threshold is applied.

    Returns:
    pd.DataFrame: The updated nodes dataframe with new demand values
    """
    # Load the CSV files
    links_df = pd.read_csv(links_path)
    nodes_df = pd.read_csv(nodes_path)

    # Create a copy of nodes_df to store updated demand values
    updated_nodes_df = nodes_df.copy()
    # Initialize the demand values to 0 for all nodes
    updated_nodes_df['demand'] = 0

    # Get all 'cn' nodes
    cn_nodes = nodes_df[nodes_df['node_type'] == 'cn']

    # Dictionary to track the current demand for each tx node
    current_tx_demand = {node_id: 0 for node_id in nodes_df['node_id']}

    # For each 'cn' node, find the appropriate tx based on path loss and current load
    for _, cn_node in cn_nodes.iterrows():
        cn_id = cn_node['node_id']
        cn_demand = cn_node['demand']

        # Get all links where the current cn is the receiver
        cn_rx_links = links_df[links_df['rx_id'] == cn_id]

        if not cn_rx_links.empty:
            # Sort links by path loss (ascending)
            sorted_links = cn_rx_links.sort_values(by='path_loss_db')

            # If no threshold is set, just pick the minimum path loss link
            if demand_threshold is None:
                min_loss_link = sorted_links.iloc[0]
                tx_id = min_loss_link['tx_id']
            else:
                # Try to find a tx that won't exceed the threshold after adding this cn's demand
                tx_chosen = False
                for _, link in sorted_links.iterrows():
                    tx_id = link['tx_id']
                    if current_tx_demand[tx_id] + cn_demand <= demand_threshold:
                        tx_chosen = True
                        break

                # If all txs would exceed threshold, pick the one with the lowest current demand
                if not tx_chosen:
                    # Find tx with minimum current demand among the available links
                    tx_id = sorted_links['tx_id'].iloc[0]  # Default to min path loss
                    min_current_demand = current_tx_demand[tx_id]

                    for _, link in sorted_links.iterrows():
                        potential_tx_id = link['tx_id']
                        if current_tx_demand[potential_tx_id] < min_current_demand:
                            tx_id = potential_tx_id
                            min_current_demand = current_tx_demand[potential_tx_id]

            # Update the demand of the chosen tx
            current_tx_demand[tx_id] += cn_demand
            updated_nodes_df.loc[updated_nodes_df['node_id'] == tx_id, 'demand'] += cn_demand

    # Save the updated nodes dataframe to a new CSV file
    updated_nodes_df.to_csv(output_path, index=False)
    print(f"Updated nodes data saved to {output_path}")

    # Return the updated nodes dataframe
    return updated_nodes_df

# Example usage
# updated_nodes = process_network_data('links.csv', 'nodes.csv', 'updated_nodes.csv', demand_threshold=100)