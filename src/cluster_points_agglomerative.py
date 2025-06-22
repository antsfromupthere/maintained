def cluster_points_agglomerative(input_path, save_path, max_distance=1000):
    """
    Perform agglomerative clustering on geographic points and save the results.

    Parameters:
    -----------
    input_path : str
        Path to the input shapefile containing points to be clustered
    save_path : str
        Path where the output shapefile with clustered points will be saved
    max_distance : float, default=1000
        Maximum distance threshold for clustering (in meters)

    Returns:
    --------
    gdf_clusters : GeoDataFrame
        The resulting GeoDataFrame with clustered points
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import matplotlib.pyplot as plt
    import contextily as ctx
    import pyproj
    import warnings

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load GeoDataFrame
    gdf = gpd.read_file(input_path)

    # Project to metric CRS (Web Mercator)
    gdf_projected = gdf.to_crs(epsg=3857)

    # Extract coordinates
    coords = np.array([(geom.x, geom.y) for geom in gdf_projected.geometry])

    # Perform clustering
    db = AgglomerativeClustering(
        distance_threshold=max_distance,
        n_clusters=None
    ).fit(coords)

    # Add cluster labels to GeoDataFrame
    gdf_projected["cluster"] = db.labels_

    # Aggregate sum of 'sau_genera' for each cluster
    clustered = (
        gdf_projected
        .groupby("cluster")
        .agg({'sau_genera': 'sum'})
        .reset_index()
        # .rename(columns={'sau_genera': 'demand'})  # Rename column here
    )

    clustered.columns = ['cluster', 'demand']  # Rename columns directly

    # Calculate the centroid of each cluster
    cluster_centroids = gdf_projected.groupby("cluster").geometry.agg(
        lambda g: Point(g.x.mean(), g.y.mean())
    )

    # Create GeoDataFrame with cluster heads (as centroids)
    gdf_clusters = gpd.GeoDataFrame(
        clustered,
        geometry=cluster_centroids,
        crs=gdf_projected.crs
    )

    # Convert back to WGS84
    gdf_clusters = gdf_clusters.to_crs(epsg=4326)

    # Add longitude/latitude columns
    gdf_clusters['longitude'] = gdf_clusters.geometry.x
    gdf_clusters['latitude'] = gdf_clusters.geometry.y

    # Reorder columns
    gdf_clusters = gdf_clusters[['longitude', 'latitude', 'demand', 'geometry']]

    # Plotting
    gdf_web = gdf.to_crs(epsg=3857)
    gdf_clusters_web = gdf_clusters.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_web.plot(ax=ax, color='gray', markersize=15, alpha=0.5, label='Original Points')
    gdf_clusters_web.plot(ax=ax, color='red', markersize=10, label='Cluster Heads')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Transform tick labels from Web Mercator to longitude/latitude
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Use the midpoint of the current axes as a reference for conversion
    mid_y = np.mean(ax.get_ylim())
    mid_x = np.mean(ax.get_xlim())

    # Convert x-axis ticks (for longitude)
    xticks = ax.get_xticks()
    new_xticks = [transformer.transform(x, mid_y)[0] for x in xticks]
    ax.set_xticklabels([f"{lon:.2f}" for lon in new_xticks])

    # Convert y-axis ticks (for latitude)
    yticks = ax.get_yticks()
    new_yticks = [transformer.transform(mid_x, y)[1] for y in yticks]
    ax.set_yticklabels([f"{lat:.2f}" for lat in new_yticks])

    # Add axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_title(f"Cluster Heads", fontsize=14)
    # ax.axis('off')
    plt.tight_layout()
    ax.legend()
    plt.show()

    # Save to shapefile
    gdf_clusters.to_file(save_path)

    return gdf_clusters
