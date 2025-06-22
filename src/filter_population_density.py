def filter_population_density(csv_file, save_path=None, north=21.5, south=21.0, east=44.0, west=43.5):
    """
    Filter population density data by geographic bounds and save to shapefile.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing population data with longitude, latitude, and sau_general_2020 columns
    save_path : str
        Path where the output shapefile will be saved
    north : float, default=21.5
        Northern boundary latitude
    south : float, default=21.0
        Southern boundary latitude
    east : float, default=44.0
        Eastern boundary longitude
    west : float, default=43.5
        Western boundary longitude

    Returns:
    --------
    gdf_filtered : GeoDataFrame
        The filtered GeoDataFrame containing points within the specified bounds
    """
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx
    import numpy as np
    import pyproj
    import warnings

    # Set matplotlib parameters for consistent styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 13
    })

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Read CSV into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Drop rows with missing coordinates or population values
    df.dropna(subset=["longitude", "latitude", "sau_general_2020"], inplace=True)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"  # WGS84 lat/lon
    )

    # Filter the GeoDataFrame based on coordinates
    gdf_filtered = gdf[
        (gdf.geometry.y >= south) & (gdf.geometry.y <= north) &
        (gdf.geometry.x >= west) & (gdf.geometry.x <= east)
        ]

    # Project to Web Mercator for visualization
    gdf_filtered_3857 = gdf_filtered.to_crs(epsg=3857)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_filtered_3857.plot(
        ax=ax,
        column="sau_general_2020",
        cmap="OrRd",
        legend=True,
        alpha=0.7,
        markersize=30
    )

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    # Convert tick labels from EPSG:3857 (meters) to EPSG:4326 (lon/lat)
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


    # Add title
    plt.title("Population Density Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


    plt.savefig('./figures/meta.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    # Save to shapefile
    if save_path is not None:
        gdf_filtered.to_file(save_path)

    return gdf_filtered

# Example usage:
# filtered_data = filter_population_density(
#     "./hrpdm/sau_general_2020.csv",
#     "./ex_data/pop_density/saudi_area_1.shp",
#     north=21.5,
#     south=21.0,
#     east=44.0,
#     west=43.5
# )