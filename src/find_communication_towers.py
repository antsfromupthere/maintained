def find_communication_towers(output_path, north=21.5, south=21.0, east=44.0, west=43.5,
                              title_fontsize=16, legend_fontsize=12, label_fontsize=14,
                              tick_fontsize=10
                              ):
    """
    Find communication towers within a specified geographic boundary and save to shapefile.
    If no towers are found, creates a single point at the center of the region.

    Parameters:
    -----------
    output_path : str
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
    combined_gdf : GeoDataFrame
        GeoDataFrame containing communication towers or center point if none found
    """
    import numpy as np
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from shapely.geometry import box, Point
    import contextily as ctx
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

    warnings.filterwarnings("ignore")

    # Define bounding box and convert to polygon
    polygon = box(west, south, east, north)

    # List of tag dictionaries for different queries
    tags_list = [
        {"man_made": "mast", "tower:type": "communication"},
        {"man_made": "tower", "tower:type": "communication", "tower:construction": "lattice"},
        {"man_made": "communications_tower"}
    ]

    # Query features within the polygon for each tag set
    gdfs = []
    for tags in tags_list:
        try:
            gdf = ox.features_from_polygon(polygon, tags)
            if not gdf.empty:
                gdfs.append(gdf)
            else:
                print(f"No features found for tags: {tags}")
        except Exception as e:
            print(f"Query for tags {tags} failed with error: {e}")

    # Process results
    if gdfs:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        # Drop duplicates based on osmid if available, else use geometry
        if 'osmid' in combined_gdf.columns:
            combined_gdf = combined_gdf.drop_duplicates(subset='osmid', keep='first')
        else:
            combined_gdf = combined_gdf.drop_duplicates(subset='geometry', keep='first')

        # Add node_type column for OSM data
        combined_gdf['pop_type'] = 'tower'
        combined_gdf['demand'] = 0
        # Ensure we have longitude and latitude columns
        combined_gdf['longitude'] = combined_gdf.geometry.x
        combined_gdf['latitude'] = combined_gdf.geometry.y

        combined_gdf.reset_index(drop=True, inplace=True)
    else:
        print("No communication towers found. Creating center point.")
        # Create a point at the center of the region
        center_lon = (east + west) / 2
        center_lat = (north + south) / 2
        # Create the GeoDataFrame with the format shown in the image
        combined_gdf = gpd.GeoDataFrame(
            {
                'longitude': center_lon,
                'latitude': center_lat,
                'pop_type': 'hap',
                'demand': 0
            },
            geometry=Point(center_lon, center_lat),
            crs="EPSG:4326"
        )

    # Ensure CRS is set properly
    combined_gdf = combined_gdf.set_crs("EPSG:4326", allow_override=True)
    combined_gdf_proj = combined_gdf.to_crs(epsg=3857)

    # Reproject the region polygon to EPSG:3857
    polygon_proj = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # Compute centroids in projected coordinates
    combined_gdf_proj['centroid'] = combined_gdf_proj.geometry.centroid
    combined_gdf_proj['lon'] = combined_gdf_proj.centroid.x
    combined_gdf_proj['lat'] = combined_gdf_proj.centroid.y

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the region boundary
    if polygon_proj.geom_type == 'Polygon':
        x, y = polygon_proj.exterior.xy
        ax.plot(x, y, color='black', linewidth=2, label='Region Boundary')
    elif polygon_proj.geom_type == 'MultiPolygon':
        for poly in polygon_proj.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=2, label='Region Boundary')

    # Plot the centroid points as red markers
    if gdfs:
        ax.scatter(combined_gdf_proj['lon'], combined_gdf_proj['lat'], color='red', s=50,
                   label='Existing Communication Tower')
    else:
        ax.scatter(combined_gdf_proj['lon'], combined_gdf_proj['lat'], color='red', s=50,
                   label='New Added HAP-Based PoP')

    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)

    # Create custom legend handles
    # region_handle = mlines.Line2D([], [], color='black', linewidth=2, label='Existing Communication Tower')
    # centroid_handle = mlines.Line2D([], [], marker='o', color='red', markersize=10,
    #                                 linestyle='None', label='Centroid Points')
    # Create legend with custom font size
    ax.legend(fontsize=legend_fontsize)

    # Set title with custom font size
    ax.set_title("Region Boundary and Communication Tower Centroids", fontsize=title_fontsize)

    # Set axis labels with custom font size
    ax.set_xlabel("Longitude", fontsize=label_fontsize)
    ax.set_ylabel("Latitude", fontsize=label_fontsize)

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

    plt.savefig('./figures/osm.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    # print("Centroid coordinates:")
    # print(combined_gdf_proj[['lon', 'lat']])

    # Save to shapefile
    combined_gdf.to_file(output_path)

    return combined_gdf

# Example usage:
# towers = find_communication_towers(
#     "./ex_data/tower/saudi_area_1.shp",
#     north=21.5,
#     south=21.0,
#     east=44.0,
#     west=43.5
# )