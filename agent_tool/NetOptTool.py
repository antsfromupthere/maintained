from smolagents import Tool
import os
import pandas as pd
from typing import Dict, Any

# Import all your existing functions
from maintained.src.enhenced_net_link_filtering import filter_network_links
from maintained.src.optimize_network_deployment_improved import optimize_network_deployment_improved

class NetworkOptimizationTool(Tool):
    """Tool for filtering links and optimizing network deployment."""

    name = "NetworkOptimizationTool"
    description = """
    Filters links and optimizes network deployment based on various criteria.
    This tool takes file paths to network data and parameters to filter links based on capacity,
    distance, and other criteria, then optimizes the deployment to minimize cost
    while maximizing coverage. It calculates the optimal cost and data rate for
    the network deployment.
    """
    inputs = {
        "netana_node_path": {
            "type": "string",
            "description": "Path to the CSV file containing node information"
        },
        "netana_link_path": {
            "type": "string",
            "description": "Path to the CSV file containing link information"
        },
        "bandwidth_Hz": {
            "type": "object",
            "description": "Bandwidth with units (from NetworkAnalysisTool)"
        },
        "tx_power_hap": {
            "type": "number",
            "description": "Transmitting power for HAP nodes in dBm (default: 20.0)",
            "nullable": False
        },
        "tx_power_tbs": {
            "type": "number",
            "description": "Transmitting power for TBS nodes in dBm (default: 20.0)",
            "nullable": False
        },
        "tx_power_pop": {
            "type": "number",
            "description": "Transmitting power for POP nodes in dBm (default: 20.0)",
            "nullable": False
        },
        "hap_cost": {
            "type": "number",
            "description": "Cost of deploying a HAP (default: 1000)",
            "nullable": False
        },
        "tbs_cost": {
            "type": "number",
            "description": "Cost of deploying a TBS (default: 500)",
            "nullable": False
        },
        "data_rate_req_mbps": {
            "type": "number",
            "description": "Minimum required data rate per demand node in Mbps (default: 2)",
            "nullable": False
        }
    }
    output_type = "object"

    def forward(self, netana_node_path: str, netana_link_path: str,
                bandwidth_Hz: Any,
                tx_power_hap: float = 20.0, tx_power_tbs: float = 20.0, tx_power_pop: float = 20.0,
                hap_cost: int = 1000, tbs_cost: int = 500,
                data_rate_req_mbps: float = 2.0) -> Dict[str, Any]:
        """
        Filter links and optimize network deployment.

        Args:
            netana_node_path: Path to the CSV file containing node information
            netana_link_path: Path to the CSV file containing link information
            bandwidth_Hz: Bandwidth with units
            tx_power_hap: Transmitting power for HAP nodes in dBm
            tx_power_tbs: Transmitting power for TBS nodes in dBm
            tx_power_pop: Transmitting power for POP nodes in dBm
            hap_cost: Cost of deploying a HAP
            tbs_cost: Cost of deploying a TBS
            data_rate_req_mbps: Minimum required data rate per demand node

        Returns:
            Dictionary containing optimization results including deployed nodes,
            active links, total cost, and achieved data rate
        """
        # Ensure output directories exist
        os.makedirs('./optimization_results', exist_ok=True)
        os.makedirs('./tmp_data/network', exist_ok=True)

        # Define power settings by node type using input parameters
        tx_power_dict = {
            'hap': tx_power_hap,
            'tbs': tx_power_tbs,
            'pop': tx_power_pop
        }

        # Define the allowed link types for backhaul network
        allowed_link_types = ['hap-tbs', 'tbs-tbs', 'pop-tbs']

        # Load nodes data
        nodes_df = pd.read_csv(netana_node_path)
        # print(f"Loaded {len(nodes_df)} nodes from {opt_node_path}")

        # Load links data
        links_df = pd.read_csv(netana_link_path)
        # print(f"Loaded {len(links_df)} links from {netana_link_path}")


        # Define output path for filtered links
        filtered_links_path = "./tmp_data/network/filtered_links.csv"

        # Run the enhanced filtering algorithm
        # print("\nRunning enhanced network filtering...")
        filtered_links_df = filter_network_links(
            nodes_df,
            links_df,
            allowed_link_types=allowed_link_types,
            max_links_per_direction=3,  # Allow up to 3 links per direction
            min_capacity_mbps=1.0,  # Minimum capacity threshold
            tx_power_dict=tx_power_dict,
            max_links_per_hap=20,  # Limit HAP-TBS links per HAP
            max_hap_distance_km=250,  # Maximum coverage distance for HAPs
            apply_grid_filtering=False,  # Apply grid-based filtering for HAP coverage
            grid_size=5,  # Grid cell size for coverage analysis
            bandwidth_Hz=10e6,
            visualize=True
        )

        filtered_links_df.to_csv(filtered_links_path, index=False)

        # Optimize network deployment
        print("\nOptimizing network deployment...")
        optimization_results = optimize_network_deployment_improved(
            filtered_links_path=filtered_links_path,
            nodes_path=netana_node_path,
            output_dir="./optimization_results",
            hap_cost=hap_cost,
            tbs_cost=tbs_cost,
            alpha=0.5,
            data_rate_req_mbps=data_rate_req_mbps,
            visualize=True,
            save_results=True
        )

        # Extract key metrics for the response
        if optimization_results["status"] == "optimal":
            response = {
                "deployment_solution_path": "./optimization_results",
                "deployment_cost": optimization_results["cost"],
                "average_data_rate": optimization_results["average_data_rate"],
                "deployed_hap_count": len(optimization_results["deployed_haps"]),
                "deployed_tbs_count": len(optimization_results["deployed_tbs"]),
                "status": "success",

            }
        else:
            response = {
                "filtered_links_path": filtered_links_path,
                "optimization_results": optimization_results,
                "tx_power_settings": tx_power_dict,
                "status": optimization_results["status"],
                "message": "Optimization failed or was infeasible"
            }

        return response