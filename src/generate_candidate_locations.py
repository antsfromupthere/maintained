def generate_candidate_locations(
        input_file_path,
        boundaries=None,  # North, South, East, West boundaries
        granularity=20000,
        access_range=5e3,
        output_file_path=None,
        plot_results=False,
        plot_grid=True
):
    """
    Generate candidate locations for base stations based on demand points.

    Parameters:
    -----------
    input_file_path : str
        Path to the shapefile containing demand points
    boundaries : tuple, optional
        Tuple of (North, South, East, West) coordinates
    granularity : float, default=20000
        The interval between two adjacent candidate locations (meters)
    access_range : float, default=5000
        Maximum distance between a candidate point and any demand point (meters)
    output_file_path : str, optional
        Path to save the output shapefile
    plot_results : bool, default=False
        Whether to generate plots of the results

    Returns:
    --------
    gdf_candidates : GeoDataFrame
        A GeoDataFrame containing the candidate locations
    """
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    import numpy as np
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    from itertools import combinations
    from geopy.distance import geodesic
    from scipy.spatial import ConvexHull

    import warnings
    warnings.filterwarnings("ignore")

    # Load GeoDataFrame
    gdf = gpd.read_file(input_file_path)

    # Project to metric CRS (Web Mercator)
    gdf_projected = gdf.to_crs(epsg=3857)

    # Extract coordinates
    demand_points = np.array([(geom.x, geom.y) for geom in gdf_projected.geometry])

    x_min, x_max = demand_points[:, 0].min(), demand_points[:, 0].max()
    y_min, y_max = demand_points[:, 1].min(), demand_points[:, 1].max()

    # Store boundaries if provided
    if boundaries:
        north, south, east, west = boundaries

    # Generate grid-based candidate locations
    x = np.arange(x_min, x_max + granularity, granularity)
    y = np.arange(y_min, y_max + granularity, granularity)
    xx, yy = np.meshgrid(x, y)
    gb_cl = np.column_stack([xx.ravel(), yy.ravel()])

    # Plot grid-based candidate locations if requested
    if plot_grid:
        # For the plot, we'll use Web Mercator for the calculation and data display
        # But we'll convert the axes to show longitude/latitude
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the demand points
        ax.scatter(demand_points[:, 0], demand_points[:, 1], c='red', label='Demand Points', s=100)

        # Plot ALL grid-based candidate locations (gb_cl)
        ax.scatter(gb_cl[:, 0], gb_cl[:, 1], c='green', marker='s',
                   label='Grid-based Candidate Locations', s=50, alpha=0.7)

        # Add grid lines to show the grid structure
        for x_val in x:
            ax.axvline(x=x_val, color='lightgray', linestyle='--', alpha=0.5)
        for y_val in y:
            ax.axhline(y=y_val, color='lightgray', linestyle='--', alpha=0.5)

        # Transform tick labels from Web Mercator to longitude/latitude
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Use the midpoint of the current axes as a reference for conversion
        mid_y = np.mean(ax.get_ylim())
        mid_x = np.mean(ax.get_xlim())

        # Convert x-axis ticks to longitude
        xticks = ax.get_xticks()
        new_xticks = [transformer.transform(x, mid_y)[0] for x in xticks]
        ax.set_xticklabels([f"{lon:.2f}" for lon in new_xticks])

        # Convert y-axis ticks to latitude
        yticks = ax.get_yticks()
        new_yticks = [transformer.transform(mid_x, y)[1] for y in yticks]
        ax.set_yticklabels([f"{lat:.2f}" for lat in new_yticks])

        # Add axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set title and legend
        ax.set_title('Grid-based Candidate Locations (gb_cl) with Demand Points')
        ax.legend()

        # Add information about parameters
        info_text = f"Granularity: {granularity} meters\nTotal grid points: {len(gb_cl)}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.show()

    # Create a convex hull around demand points
    hull = ConvexHull(demand_points)
    hull_points = demand_points[hull.vertices]
    polygon = Polygon(hull_points)

    # Filter candidate points to those inside the convex hull
    inside = np.array([polygon.contains(Point(x, y)) for x, y in gb_cl])
    candidate_points = gb_cl[inside]

    # Filter candidate points based on access range
    valid = []
    for candidate in candidate_points:
        distances = np.linalg.norm(demand_points - candidate, axis=1)
        if np.any(distances <= access_range):
            valid.append(True)
        else:
            valid.append(False)

    candidate_points_filtered = candidate_points[valid]

    # Plot if requested
    # Plot if requested
    if plot_results:
        # For the plot, we'll use Web Mercator for the calculation and data display
        # But we'll convert the axes to show longitude/latitude
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the demand points and candidate points in Web Mercator
        ax.scatter(demand_points[:, 0], demand_points[:, 1], c='red', label='Demand Points', s=100)
        ax.scatter(candidate_points_filtered[:, 0], candidate_points_filtered[:, 1],
                   c='blue', marker='x', label='Candidate Points', s=50)

        # Plot the convex hull
        if polygon:
            ax.plot(*polygon.exterior.xy, 'g--', label='Convex Hull')

        # Add grid
        # ax.grid(True, linestyle='--', alpha=0.5)

        # Transform tick labels from Web Mercator to longitude/latitude
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Use the midpoint of the current axes as a reference for conversion
        mid_y = np.mean(ax.get_ylim())
        mid_x = np.mean(ax.get_xlim())

        # Convert x-axis ticks to longitude
        xticks = ax.get_xticks()
        new_xticks = [transformer.transform(x, mid_y)[0] for x in xticks]
        ax.set_xticklabels([f"{lon:.2f}" for lon in new_xticks])

        # Convert y-axis ticks to latitude
        yticks = ax.get_yticks()
        new_yticks = [transformer.transform(mid_x, y)[1] for y in yticks]
        ax.set_yticklabels([f"{lat:.2f}" for lat in new_yticks])

        # Add axis labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set title and legend
        ax.set_title('Candidate Locations with Demand Points')
        ax.legend()

        plt.tight_layout()
        plt.show()

    # Transform back to geographic coordinates (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(candidate_points_filtered[:, 0], candidate_points_filtered[:, 1])
    points_4326 = np.column_stack((lon, lat))

    # Create a GeoDataFrame
    gdf_candidates = gpd.GeoDataFrame(
        {
            'longitude': points_4326[:, 0],
            'latitude': points_4326[:, 1],
            'geometry': [Point(lon, lat) for lon, lat in points_4326],
            'demand': 0
        },
        crs="EPSG:4326"
    )

    # Calculate distance statistics
    coordinates = [(point.y, point.x) for point in gdf_candidates.geometry]

    if len(coordinates) > 1:  # Only compute distances if there are at least 2 points
        distances = []
        for (lat1, lon1), (lat2, lon2) in combinations(coordinates, 2):
            dist = geodesic((lat1, lon1), (lat2, lon2)).meters
            distances.append(dist)

        min_distance = np.min(distances) if distances else 0
        max_distance = np.max(distances) if distances else 0
        mean_distance = np.mean(distances) if distances else 0
        median_distance = np.median(distances) if distances else 0

        # print(f"Minimum geodesic distance: {min_distance:.2f} meters")
        # print(f"Maximum geodesic distance: {max_distance:.2f} meters")
        # print(f"Mean geodesic distance: {mean_distance:.2f} meters")
        # print(f"Median geodesic distance: {median_distance:.2f} meters")

    # Save to file if path is provided
    if output_file_path:
        gdf_candidates.to_file(output_file_path)

    return gdf_candidates