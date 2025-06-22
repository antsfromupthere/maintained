import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from maintained.src.plot_deployment_decision import visualize_connectivity_3d

def calculate_data_rate_bounds(df_links, df_nodes, node_types, node_demands,
                               link_caps, incoming_links, demand_nodes):
    """
    Calculate theoretical maximum and minimum data rates for normalization.

    Returns:
    --------
    tuple: (R_min, R_max)
    """

    # Method 1: Calculate R_max as sum of all incoming capacities to demand nodes
    R_max_method1 = 0
    for node in demand_nodes:
        incoming_cap = sum(link_caps[u, v] for u, v in incoming_links[node])
        R_max_method1 += incoming_cap

    # Method 2: Calculate R_max as total network capacity
    R_max_method2 = sum(link_caps.values())

    # Method 3: Calculate R_max based on bottleneck analysis
    R_max_method3 = 0
    for node in demand_nodes:
        # Find maximum flow that can reach this node
        incoming_cap = sum(link_caps[u, v] for u, v in incoming_links[node])
        R_max_method3 += min(incoming_cap, sum(node_demands.values()))

    # Choose the most conservative (realistic) estimate
    R_max = R_max_method1  # Most realistic upper bound

    # Calculate R_min as the minimum required data rate
    R_min = sum(max(node_demands[node], 2) for node in demand_nodes)  # At least 2 Mbps per node

    print(f"Data rate bounds calculation:")
    print(f"  R_min (minimum required): {R_min:.2f} Mbps")
    print(f"  R_max (theoretical maximum): {R_max:.2f} Mbps")
    print(f"  R_max/R_min ratio: {R_max / R_min:.2f}")

    return R_min, R_max


def calculate_cost_bounds(hap_nodes, tbs_nodes, hap_cost, tbs_cost,
                          pop_nodes, demand_nodes):
    """
    Calculate cost bounds for normalization.

    Returns:
    --------
    tuple: (C_min, C_max)
    """

    # C_max: Deploy all possible nodes
    C_max = len(hap_nodes) * hap_cost + len(tbs_nodes) * tbs_cost

    # C_min: Deploy only essential nodes (POPs are free, demand nodes must be deployed)
    essential_haps = [n for n in demand_nodes if n in hap_nodes]
    essential_tbs = [n for n in demand_nodes if n in tbs_nodes]

    C_min = len(essential_haps) * hap_cost + len(essential_tbs) * tbs_cost
    return C_min, C_max


def setup_objective_function(solver, method='minmax_norm', alpha=0.5,
                             is_deployed=None, node_data_rate=None,
                             hap_nodes=None, tbs_nodes=None, demand_nodes=None,
                             hap_cost=1000, tbs_cost=500,
                             C_min=0, C_max=10000, R_min=100, R_max=1000,
                             candidate_nodes=None):
    """
    Setup different objective function formulations to handle the scaling issue.

    Parameters:
    -----------
    method : str
        'minmax_norm' : Min-max normalization
        'reciprocal' : Use reciprocal transformation
        'balanced_weights' : Use balanced weight scaling
        'goal_programming' : Goal programming approach
    """

    objective = solver.Objective()

    if method == 'minmax_norm':
        # Method 1: Min-Max Normalization (both terms in [0,1])
        print("Using Min-Max Normalization")

        # Cost component: (C - C_min) / (C_max - C_min)
        if C_max > C_min:
            for node in hap_nodes:
                if node in candidate_nodes:
                    cost_coeff = alpha * hap_cost / (C_max - C_min)
                    objective.SetCoefficient(is_deployed[node], cost_coeff)

            for node in tbs_nodes:
                if node in candidate_nodes:
                    cost_coeff = alpha * tbs_cost / (C_max - C_min)
                    objective.SetCoefficient(is_deployed[node], cost_coeff)

        # Data rate component: -(1-α) * (R - R_min) / (R_max - R_min)
        if R_max > R_min:
            for node in demand_nodes:
                rate_coeff = -(1 - alpha) / (R_max - R_min)
                objective.SetCoefficient(node_data_rate[node], rate_coeff)

    elif method == 'reciprocal':
        # Method 2: Reciprocal Transformation
        print("Using Reciprocal Transformation")

        # Cost component: α * C / C_max
        for node in hap_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * hap_cost / C_max
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        for node in tbs_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * tbs_cost / C_max
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        # Rate component: -(1-α) * R / R_target
        R_target = (R_min + R_max) / 2  # Use midpoint as target
        for node in demand_nodes:
            rate_coeff = -(1 - alpha) / R_target
            objective.SetCoefficient(node_data_rate[node], rate_coeff)

    elif method == 'balanced_weights':
        # Method 3: Balanced Weight Scaling
        print("Using Balanced Weight Scaling")

        # Calculate scale factors to balance the magnitude of cost and rate terms
        cost_scale = C_max / len(candidate_nodes) if len(candidate_nodes) > 0 else 1
        rate_scale = R_max / len(demand_nodes) if len(demand_nodes) > 0 else 1

        # Balance factor to make terms comparable
        balance_factor = cost_scale / rate_scale if rate_scale > 0 else 1

        # Cost component
        for node in hap_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * hap_cost / cost_scale
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        for node in tbs_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * tbs_cost / cost_scale
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        # Rate component (with balance factor)
        for node in demand_nodes:
            rate_coeff = -(1 - alpha) * balance_factor / rate_scale
            objective.SetCoefficient(node_data_rate[node], rate_coeff)

    elif method == 'goal_programming':
        # Method 4: Goal Programming
        print("Using Goal Programming")

        # Set target values
        C_target = (C_min + C_max) / 2  # Target cost
        R_target = (R_min + R_max) * 0.8  # Target 80% of max rate

        print(f"  Cost target: {C_target}")
        print(f"  Rate target: {R_target}")

        # Cost component: weight * (C - C_target) / C_target
        for node in hap_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * hap_cost / C_target
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        for node in tbs_nodes:
            if node in candidate_nodes:
                cost_coeff = alpha * tbs_cost / C_target
                objective.SetCoefficient(is_deployed[node], cost_coeff)

        # Rate component: -weight * (R - R_target) / R_target
        for node in demand_nodes:
            rate_coeff = -(1 - alpha) / R_target
            objective.SetCoefficient(node_data_rate[node], rate_coeff)

    else:
        raise ValueError(f"Unknown objective method: {method}")

    objective.SetMinimization()

    return objective


def optimize_network_deployment_improved(
        filtered_links_path,
        nodes_path,
        output_dir='./optimization_results',
        hap_cost=1000,
        tbs_cost=500,
        alpha=0.5,
        objective_method='minmax_norm',  # New parameter
        data_rate_req_mbps=2,
        visualize=True,
        save_results=True
):
    """
    Optimize network deployment using a weighted sum of deployment cost and data rate.
    Improved version with better bounds calculation and multiple objective formulations.

    Parameters:
    -----------
    filtered_links_path : str
        Path to the CSV file containing filtered network links
    nodes_path : str
        Path to the CSV file containing node information
    output_dir : str, default='./optimization_results'
        Directory to save output files
    hap_cost : int, default=1000
        Cost to deploy a HAP node
    tbs_cost : int, default=500
        Cost to deploy a TBS node
    alpha : float, default=0.5
        Weight factor for the objective function (0 ≤ alpha ≤ 1)
        alpha = 1: Only consider cost minimization
        alpha = 0: Only consider data rate maximization
    objective_method : str, default='minmax_norm'
        Method for objective function: 'minmax_norm', 'reciprocal', 'balanced_weights', 'goal_programming'
    data_rate_req_mbps : float, default=2
        Minimum data rate requirement per demand node
    visualize : bool, default=True
        Whether to generate and display a visualization
    save_results : bool, default=True
        Whether to save results to files

    Returns:
    --------
    dict
        Dictionary containing optimization results including deployed nodes,
        active links, total cost, and achieved data rate
    """
    # Create output directory if it doesn't exist
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    df_links = pd.read_csv(filtered_links_path)
    df_nodes = pd.read_csv(nodes_path)

    # Create lookups
    node_types = dict(zip(df_nodes['node_id'], df_nodes['node_type']))
    node_demands = dict(zip(df_nodes['node_id'], df_nodes['demand']))

    # Identify nodes by type
    pop_nodes = [n for n, t in node_types.items() if t.lower() == 'pop']
    hap_nodes = [n for n, t in node_types.items() if t.lower() == 'hap']
    tbs_nodes = [n for n, t in node_types.items() if t.lower() == 'tbs']

    # Filter out HAPs from having demand
    for node_id in hap_nodes:
        if node_id in node_demands and node_demands[node_id] > 0:
            node_demands[node_id] = 0

    # Identify nodes with demand (excluding POPs)
    demand_nodes = [n for n, d in node_demands.items() if d > 0 and node_types.get(n, '').lower() != 'pop']

    # Get valid nodes that appear in the filtered links
    network_nodes = set()
    for _, link in df_links.iterrows():
        tx_id, rx_id = link['tx_id'], link['rx_id']
        if tx_id in node_types and node_types[tx_id].lower() in ['tbs', 'hap', 'pop']:
            network_nodes.add(tx_id)
        if rx_id in node_types and node_types[rx_id].lower() in ['tbs', 'hap', 'pop']:
            network_nodes.add(rx_id)

    # HAP and TBS nodes are candidates for deployment decision
    candidate_nodes = [n for n in network_nodes if n in node_types and
                       node_types[n].lower() in ['hap', 'tbs']]

    # Create link capacity dictionary
    link_caps = {(link['tx_id'], link['rx_id']): link['cap_mbps'] for _, link in df_links.iterrows()}

    # Create incoming/outgoing links dictionaries
    incoming_links = collections.defaultdict(list)
    outgoing_links = collections.defaultdict(list)
    for u, v in link_caps:
        outgoing_links[u].append((u, v))
        incoming_links[v].append((u, v))

    # Delete nodes with no incoming links from demand_nodes
    nodes_without_incoming = []
    for node in demand_nodes[:]:  # Iterate over a copy
        if len(incoming_links[node]) < 1:
            nodes_without_incoming.append(node)
            demand_nodes.remove(node)

    # Check demand feasibility
    infeasible_demand_nodes = []
    for node_index in demand_nodes[:]:  # Iterate over a copy
        incoming_cap = sum(link_caps[u, v] for u, v in incoming_links[node_index])
        if incoming_cap <= node_demands[node_index]:
            infeasible_demand_nodes.append((node_index, node_demands[node_index], incoming_cap))
            node_demands[node_index] = 0.0

    # Calculate proper bounds
    R_min, R_max = calculate_data_rate_bounds(
        df_links, df_nodes, node_types, node_demands,
        link_caps, incoming_links, demand_nodes
    )

    C_min, C_max = calculate_cost_bounds(
        hap_nodes, tbs_nodes, hap_cost, tbs_cost,
        pop_nodes, demand_nodes
    )

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print("No solver available")
            return {"status": "error", "message": "No solver available"}

    # Create variables
    flow = {}  # Flow on each link
    tdm = {}  # Time division multiplexing fraction
    for u, v in link_caps:
        flow[u, v] = solver.NumVar(0, link_caps[u, v], f'flow_{u}_{v}')
        tdm[u, v] = solver.NumVar(0, 1, f'tdm_{u}_{v}')

    # Node deployment decision variables
    is_deployed = {}
    for node in candidate_nodes:
        is_deployed[node] = solver.BoolVar(f'deploy_{node}')

    # Calculate a reasonable Big-M value
    big_M = sum(node_demands.values()) if node_demands else sum(link_caps.values())
    if big_M == 0:
        big_M = 1e9  # Fallback value

    # Variables to track the data rate at each demand node
    node_data_rate = {}
    for node in demand_nodes:
        node_data_rate[node] = solver.NumVar(0, big_M, f'data_rate_{node}')

    # 1. Flow conservation constraints
    for node_index in network_nodes:
        inflow = solver.Sum(flow[u, v] for u, v in incoming_links[node_index])
        outflow = solver.Sum(flow[u, v] for u, v in outgoing_links[node_index])

        if node_index in demand_nodes:
            # Demand nodes must receive at least the minimum required data rate
            solver.Add(inflow - outflow >= data_rate_req_mbps, f'min_demand_{node_index}')

            # Track the actual data rate at this demand node
            solver.Add(node_data_rate[node_index] == inflow - outflow, f'actual_rate_{node_index}')
        else:
            if node_types[node_index].lower() != 'hap' and node_types[node_index].lower() != 'pop':
                # Transit nodes maintain flow balance
                solver.Add(inflow == outflow, f'balance_{node_index}')

    # 2. TDM constraints
    for node_index in network_nodes:
        # Incoming TDM fractions must sum to <= 1
        solver.Add(solver.Sum(tdm[u, v] for u, v in incoming_links[node_index]) <= 1,
                   f'tdm_in_{node_index}')
        # Outgoing TDM fractions must sum to <= 1
        solver.Add(solver.Sum(tdm[u, v] for u, v in outgoing_links[node_index]) <= 1,
                   f'tdm_out_{node_index}')

    # 3. Link capacity constraints
    for u, v in link_caps:
        solver.Add(flow[u, v] <= tdm[u, v] * link_caps[u, v], f'cap_{u}_{v}')

    # 4. Node deployment constraints
    # POPs are always deployed
    for node in pop_nodes:
        if node in candidate_nodes:
            solver.Add(is_deployed[node] == 1, f'pop_always_{node}')

    # Demand nodes must be deployed
    for node in demand_nodes:
        if node in candidate_nodes:
            solver.Add(is_deployed[node] == 1, f'demand_node_{node}')

    # Flow through a node only if it's deployed
    for node in candidate_nodes:
        inflow = solver.Sum(flow[u, v] for u, v in incoming_links[node])
        outflow = solver.Sum(flow[u, v] for u, v in outgoing_links[node])

        solver.Add(inflow <= big_M * is_deployed[node], f'flow_in_{node}')
        solver.Add(outflow <= big_M * is_deployed[node], f'flow_out_{node}')

    # Setup objective function with the chosen method
    objective = setup_objective_function(
        solver,
        method=objective_method,
        alpha=alpha,
        is_deployed=is_deployed,
        node_data_rate=node_data_rate,
        hap_nodes=hap_nodes,
        tbs_nodes=tbs_nodes,
        demand_nodes=demand_nodes,
        hap_cost=hap_cost,
        tbs_cost=tbs_cost,
        C_min=C_min,
        C_max=C_max,
        R_min=R_min,
        R_max=R_max,
        candidate_nodes=candidate_nodes
    )

    # Solve the problem
    print("\nSolving optimization problem...")
    status = solver.Solve()

    # Process results
    results = {}

    if status == pywraplp.Solver.OPTIMAL:
        # Calculate the actual values
        actual_cost = 0
        for node in candidate_nodes:
            if is_deployed[node].solution_value() > 0.5:
                if node in hap_nodes:
                    actual_cost += hap_cost
                elif node in tbs_nodes:
                    actual_cost += tbs_cost

        # Calculate actual data rate
        actual_total_data_rate = sum(node_data_rate[node].solution_value() for node in demand_nodes)
        average_data_rate = actual_total_data_rate / len(demand_nodes) if demand_nodes else 0

        # Calculate normalized values based on the objective method
        if objective_method == 'minmax_norm':
            normalized_cost = (actual_cost - C_min) / (C_max - C_min) if C_max > C_min else 0
            normalized_rate = (actual_total_data_rate - R_min) / (R_max - R_min) if R_max > R_min else 0
            obj_value = alpha * normalized_cost - (1 - alpha) * normalized_rate
        elif objective_method == 'reciprocal':
            normalized_cost = actual_cost / C_max if C_max > 0 else 0
            R_target = (R_min + R_max) / 2
            normalized_rate = actual_total_data_rate / R_target if R_target > 0 else 0
            obj_value = alpha * normalized_cost - (1 - alpha) * normalized_rate
        elif objective_method == 'balanced_weights':
            cost_scale = C_max / len(candidate_nodes) if len(candidate_nodes) > 0 else 1
            rate_scale = R_max / len(demand_nodes) if len(demand_nodes) > 0 else 1
            balance_factor = cost_scale / rate_scale if rate_scale > 0 else 1
            normalized_cost = actual_cost / cost_scale if cost_scale > 0 else 0
            normalized_rate = actual_total_data_rate * balance_factor / rate_scale if rate_scale > 0 else 0
            obj_value = alpha * normalized_cost - (1 - alpha) * normalized_rate
        elif objective_method == 'goal_programming':
            C_target = (C_min + C_max) / 2
            R_target = (R_min + R_max) * 0.8
            normalized_cost = actual_cost / C_target if C_target > 0 else 0
            normalized_rate = actual_total_data_rate / R_target if R_target > 0 else 0
            obj_value = alpha * normalized_cost - (1 - alpha) * normalized_rate
        else:
            normalized_cost = actual_cost / C_max if C_max > 0 else 0
            normalized_rate = actual_total_data_rate / R_max if R_max > 0 else 0
            obj_value = alpha * normalized_cost - (1 - alpha) * normalized_rate

        print(f"Optimal solution found:")
        print(f"- Objective method: {objective_method}")
        print(f"- Deployment cost: {actual_cost} (normalized: {normalized_cost:.4f})")
        print(f"- Total data rate: {actual_total_data_rate:.2f} Mbps (normalized: {normalized_rate:.4f})")
        print(f"- Average data rate per demand node: {average_data_rate:.2f} Mbps")
        print(f"- Objective value: {obj_value:.4f}")

        # Get deployed nodes
        deployed_nodes = []
        for node in candidate_nodes:
            if is_deployed[node].solution_value() > 0.5:  # Binary variable ~1
                deployed_nodes.append(node)

        # Add POP nodes (always deployed)
        all_deployed = deployed_nodes + [n for n in pop_nodes if n not in deployed_nodes]

        # Get active links
        active_links = []
        for u, v in link_caps:
            flow_val = flow[u, v].solution_value()
            if flow_val > 1e-6:  # Flow greater than epsilon
                active_links.append({
                    'tx_id': u,
                    'rx_id': v,
                    'flow_mbps': flow_val,
                    'tdm': tdm[u, v].solution_value(),
                    'capacity': link_caps[u, v],
                    'utilization': flow_val / link_caps[u, v]
                })

        # Get data rates for each demand node
        demand_node_rates = {node: node_data_rate[node].solution_value() for node in demand_nodes}
        demand_rates_df = pd.DataFrame({
            'node_id': list(demand_node_rates.keys()),
            'data_rate_mbps': list(demand_node_rates.values())
        })

        # Save deployed nodes to CSV
        deployed_df = df_nodes[df_nodes['node_id'].isin(all_deployed)]
        active_links_df = pd.DataFrame(active_links)

        if save_results:
            deployed_df.to_csv(f"{output_dir}/deployed_nodes.csv", index=False)
            active_links_df.to_csv(f"{output_dir}/active_links.csv", index=False)
            demand_rates_df.to_csv(f"{output_dir}/demand_node_rates.csv", index=False)
            print(f"Results saved to {output_dir}")

        # Print summary
        hap_count = sum(1 for n in deployed_nodes if n in hap_nodes)
        tbs_count = sum(1 for n in deployed_nodes if n in tbs_nodes)
        pop_count = len(pop_nodes)

        print(f"\nDeployment Summary:")
        print(f"Total nodes deployed: {len(all_deployed)}")
        print(f" - HAP nodes: {hap_count} (cost: {hap_count * hap_cost})")
        print(f" - TBS nodes: {tbs_count} (cost: {tbs_count * tbs_cost})")
        print(f" - POP nodes: {pop_count} (always deployed)")
        print(f"Active links: {len(active_links)}")
        print(f"Total deployment cost: {actual_cost}")
        print(f"Average data rate: {average_data_rate:.2f} Mbps")

        visualize_connectivity_3d(nodes_csv_path="./tmp_data/network//nodes.csv",
                                  deployed_nodes_csv_path=f"{output_dir}/deployed_nodes.csv",
                                  active_links_csv_path=f"{output_dir}/active_links.csv",
                                  links_csv_path="./tmp_data/network/links.csv")

        # Store results in dictionary
        results = {
            "status": "optimal",
            "objective_method": objective_method,
            "cost": actual_cost,
            "total_data_rate": actual_total_data_rate,
            "average_data_rate": average_data_rate,
            "obj_value": obj_value,
            "alpha": alpha,
            "normalized_cost": normalized_cost,
            "normalized_rate": normalized_rate,
            "cost_bounds": (C_min, C_max),
            "rate_bounds": (R_min, R_max),
            "deployed_nodes": all_deployed,
            "deployed_haps": [n for n in deployed_nodes if n in hap_nodes],
            "deployed_tbs": [n for n in deployed_nodes if n in tbs_nodes],
            "deployed_pops": pop_nodes,
            "active_links": active_links,
            "demand_node_rates": demand_node_rates,
            "deployed_nodes_df": deployed_df,
            "active_links_df": active_links_df,
            "demand_rates_df": demand_rates_df
        }

    elif status == pywraplp.Solver.INFEASIBLE:
        print("Problem is infeasible - no solution exists.")
        results = {"status": "infeasible"}
    else:
        print(f"Solver status: {status}")
        results = {"status": "unknown", "solver_status": status}

    return results


# Set matplotlib parameters for consistent styling
