from smolagents import Tool
import os
from typing import Dict

from maintained.src.find_communication_towers import find_communication_towers
from maintained.src.filter_population_density import filter_population_density
from maintained.src.cluster_points_agglomerative import cluster_points_agglomerative
from maintained.src.generate_candidate_locations import generate_candidate_locations


class GeographicDataCollectionTool(Tool):
    """Tool for collecting geographic data and generating candidate locations."""

    name = "GeographicDataCollectionTool"
    description = """
    Collects geographic data and generates candidate locations for network deployment.
    This tool handles finding communication towers, filtering population density, 
    clustering points, and generating TBS and HAP candidate locations. It returns
    file paths to the generated shapefiles for further processing.
    """
    inputs = {
        "north": {
            "type": "number",
            "description": "North boundary coordinate (latitude)"
        },
        "south": {
            "type": "number",
            "description": "South boundary coordinate (latitude)"
        },
        "east": {
            "type": "number",
            "description": "East boundary coordinate (longitude)"
        },
        "west": {
            "type": "number",
            "description": "West boundary coordinate (longitude)"
        },
        "meta_hrpdm_path": {
            "type": "string",
            "description": "Path to population density data CSV (default: './hrpdm/sau_general_2020.csv')",
            "nullable": True
        }
    }
    output_type = "object"
    # tbs_granularity: float = 4e3,
    def forward(self, north: float, south: float, east: float, west: float,
                # tbs_granularity: float = 5e3, tbs_access_range: float = 5e3,
                # hap_granularity: float = 20e3, hap_access_range: float = 15e3,
                meta_hrpdm_path: str = './hrpdm/sau_general_2020.csv') -> Dict[str, str]:
        """
        Collect geographic data and generate candidate locations.

        Args:
            north: North boundary coordinate
            south: South boundary coordinate
            east: East boundary coordinate
            west: West boundary coordinate
            meta_hrpdm_path: Path to population density data CSV

        Returns:
            Dictionary containing paths to the generated shapefiles
        """
        # Ensure output directories exist
        os.makedirs('./tmp_data/pop_locs', exist_ok=True)
        os.makedirs('./tmp_data/raw_cn_locs', exist_ok=True)
        os.makedirs('./tmp_data/cn_locs', exist_ok=True)
        os.makedirs('./tmp_data/dn_locs/tbs', exist_ok=True)
        os.makedirs('./tmp_data/dn_locs/hap', exist_ok=True)

        boundaries = (north, south, east, west)

        # Define paths
        pop_loc_path = './tmp_data/pop_locs/tower.shp'
        raw_cn_locs_path = './tmp_data/raw_cn_locs/raw.shp'
        cn_loc_path = './tmp_data/cn_locs/cn.shp'
        tbs_loc_path = './tmp_data/dn_locs/tbs/tbs.shp'
        hap_loc_path = './tmp_data/dn_locs/hap/hap.shp'

        # Step 1: Find communication towers
        print("Finding communication towers...")
        find_communication_towers(output_path=pop_loc_path)

        # Step 2: Filter population density
        print("Filtering population density...")
        filter_population_density(csv_file=meta_hrpdm_path,
                                  save_path=raw_cn_locs_path)

        # Step 3: Cluster points
        print("Clustering points...")
        cluster_points_agglomerative(input_path=raw_cn_locs_path, save_path=cn_loc_path)

        # Step 4: Generate TBS candidate locations
        print("Generating TBS candidate locations...")
        generate_candidate_locations(
            input_file_path=cn_loc_path,
            boundaries=boundaries,
            granularity=5000,
            access_range=5000,
            output_file_path=tbs_loc_path,
            plot_results=True
        )

        # Step 5: Generate HAP candidate locations
        print("Generating HAP candidate locations...")
        generate_candidate_locations(
            input_file_path=cn_loc_path,
            boundaries=boundaries,
            granularity=20000,
            access_range=15000,
            output_file_path=hap_loc_path,
            plot_results=True
        )

        print(f"Data collection and candidate location generation complete.")
        print(f"Generated the following data files:")
        print(f"- Population locations: {pop_loc_path}")
        print(f"- Raw CN locations: {raw_cn_locs_path}")
        print(f"- Clustered CN locations: {cn_loc_path}")
        print(f"- TBS candidate locations: {tbs_loc_path}")
        print(f"- HAP candidate locations: {hap_loc_path}")

        return {
            "pop_loc_path": pop_loc_path,
            "raw_cn_locs_path": raw_cn_locs_path,
            "cn_loc_path": cn_loc_path,
            "tbs_loc_path": tbs_loc_path,
            "hap_loc_path": hap_loc_path
        }