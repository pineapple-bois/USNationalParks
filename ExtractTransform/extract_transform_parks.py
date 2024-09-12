import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import re
import os
import logging
from fuzzywuzzy import process
import contextily as ctx
from shapely.geometry import Point


## master url for GeoJSON data
# https://catalog.data.gov/dataset/national-park-boundaries/resource/cee04cfe-f439-4a65-91c0-ca2199fa5f93


def setup_logger():
    log_dir = "Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'extract_transform_parks.log')
    logger = logging.getLogger('transformation_parks')
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

# Function to clean park names by removing "National Park", "National Parks", and specific variations
def clean_park_name(name):
    return re.sub(r' National Parks?$| National Park and Preserve$', '', name, flags=re.IGNORECASE).strip()

# convert acres to square kilometers
def acres_to_sq_km(acres):
    return round(acres * 0.00404686, 2)

def fetch_geojson(api_url, logger):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        file_info = response.json()
        download_url = file_info['download_url']
        gdf = gpd.read_file(download_url)
        assert gdf.shape == (510, 10), "The GeoDataFrame is not the required dimension"
        logger.info("Successfully fetched GeoJSON data from GitHub.")
        return gdf
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred: {err}")

def match_parks(df, gdf, logger):
    df['Cleaned_Park_Name'] = df['Park Name'].apply(clean_park_name)
    name_matches = {}

    for index, row in df.iterrows():
        park_code = row['Park Code']
        cleaned_name = row['Cleaned_Park_Name']
        matching_rows = gdf[gdf['UNIT_NAME'].str.contains(cleaned_name, case=False, na=False)]

        if not matching_rows.empty:
            matched_name = matching_rows['UNIT_NAME'].values[0]
            matched_code = matching_rows['UNIT_CODE'].values[0]
            name_matches[cleaned_name] = (matched_name, matched_code, park_code)
        else:
            logger.warning(f"No match found for {cleaned_name} in GeoJSON.")

    # Granular searching for unmatched parks using fuzzy matching
    unmatched_csv = [row['Cleaned_Park_Name'] for _, row in df.iterrows() if row['Cleaned_Park_Name'] not in name_matches]
    for park_name in unmatched_csv:
        best_match = process.extractOne(park_name, gdf['UNIT_NAME'], scorer=process.fuzz.partial_ratio)
        if best_match and best_match[1] > 80:
            matched_name = best_match[0]
            matched_code = gdf.loc[gdf['UNIT_NAME'] == matched_name, 'UNIT_CODE'].values[0]
            csv_code = df.loc[df['Cleaned_Park_Name'] == park_name, 'Park Code'].values[0]
            name_matches[park_name] = (matched_name, matched_code, csv_code)
            logger.info(f"Fuzzy matched {park_name} (CSV) with {matched_name} (GeoJSON) - Match score: {best_match[1]}")
        else:
            logger.warning(f"No good fuzzy match found for {park_name} in GeoJSON.")

    return name_matches

def update_geojson(gdf, name_matches, logger):
    mapping_dict = {matched_code: csv_code for _, (matched_name, matched_code, csv_code) in name_matches.items()}
    for matched_code, csv_code in mapping_dict.items():
        gdf.loc[(gdf['UNIT_CODE'] == matched_code) & (gdf['UNIT_TYPE'] == 'National Park'), 'UNIT_CODE'] = csv_code
    updated_gdf = gdf[(gdf['UNIT_CODE'].isin(mapping_dict.values())) & (gdf['UNIT_TYPE'] == 'National Park')]
    assert updated_gdf.shape == (55, 10), "The GeoDataFrame is not the required dimension"
    logger.info("GeoJSON updated with matched park codes.")
    return updated_gdf

def ensure_data_consistency(df, geo_df, logger):
    park_set_df = set(df["Park Code"])
    park_set_gdf = set(geo_df['UNIT_CODE'])
    try:
        assert park_set_df == park_set_gdf, (f"Park codes do not match: Missing in GeoDataFrame: {park_set_df - park_set_gdf}, "
                                             f"Extra in GeoDataFrame: {park_set_gdf - park_set_df}")
    except AssertionError as e:
        logger.error(e)
        raise
    else:
        logger.info("All park codes match correctly between CSV and GeoDataFrame.")

def prepare_parks_dataframe(df, logger):
    """
    Prepares the parks DataFrame by cleaning park names, converting acres to square kilometers,
    and adding necessary columns for GeoDataFrame creation.

    Args:
        df (pd.DataFrame): Initial DataFrame containing park data.
        logger: Logger for logging information and errors.

    Returns:
        pd.DataFrame: A cleaned and prepared DataFrame.
    """
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df['square_km'] = df['acres'].apply(acres_to_sq_km)
    df['park_name'] = df['park_name'].str.replace(r' National Park.*', '', case=False, regex=True)
    logger.info("Parks DataFrame prepared with square kilometers and cleaned park names.")
    return df

def create_points_geojson(prepared_df, output_dir, logger):
    """
    Creates a GeoDataFrame for point geometries from the prepared DataFrame and exports it as GeoJSON.

    Args:
        prepared_df (pd.DataFrame): DataFrame prepared with necessary columns and cleaned data.
        output_dir (str): Directory to save the GeoJSON file.
        logger: Logger for logging information and errors.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with point geometries.
    """
    prepared_df['geometry'] = prepared_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    geo_df_points = gpd.GeoDataFrame(prepared_df.drop(columns=['acres', 'latitude', 'longitude']),
                                     geometry='geometry', crs="EPSG:4326")
    export_geojson(geo_df_points, os.path.join(output_dir, "parks_points.geojson"), "parks_points.geojson", logger)
    logger.info("Points GeoDataFrame created and exported.")
    return geo_df_points, prepared_df


def merge_shape_data(prepared_df, geo_df, logger):
    """
    Merges the prepared parks DataFrame with the GeoJSON DataFrame and ensures that geometry data is handled correctly.

    Args:
        prepared_df (pd.DataFrame): DataFrame prepared with necessary columns and cleaned data.
        geo_df (gpd.GeoDataFrame): GeoDataFrame containing geometry data to be merged.
        logger: Logger for logging information and errors.

    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame with geometry data.
    """
    logger.info("Starting the merging process of parks data with GeoJSON data.")
    # Reverting the prepared_df currently populated with POINT geometry
    if 'geometry' in prepared_df.columns:
        prepared_df = prepared_df.drop(columns=['geometry'])
        logger.info("Dropped 'geometry' column from prepared_df before merging.")

    merged_parks = pd.merge(prepared_df, geo_df[['UNIT_CODE', 'geometry']],
                            left_on='park_code', right_on='UNIT_CODE', how='left')
    logger.info(f"Merged DataFrame shape: {merged_parks.shape}")

    merged_parks = merged_parks.drop(columns=['acres', 'latitude', 'longitude', 'cleaned_park_name', 'UNIT_CODE'])
    if 'geometry' not in merged_parks.columns or merged_parks['geometry'].isnull().all():
        logger.error("Geometry column is missing or contains all null values after merging. Check input data.")
        raise ValueError("Geometry column is missing or contains all null values after merging.")

    # Convert to GeoDataFrame
    merged_geo_parks = gpd.GeoDataFrame(merged_parks, geometry='geometry', crs="EPSG:4326")

    # Ensure the GeoDataFrame has the correct CRS
    if merged_geo_parks.crs != "EPSG:4326":
        merged_geo_parks = merged_geo_parks.to_crs("EPSG:4326")

    logger.info("Successfully merged parks data with GeoJSON data.")
    return merged_geo_parks

def export_geojson(geo_df, file_path, file_name, logger):
    """
    Exports a GeoDataFrame as a GeoJSON file.

    Args:
        geo_df (gpd.GeoDataFrame): The GeoDataFrame to export.
        file_path (str): The path where the GeoJSON will be saved.
        file_name (str): The name of the GeoJSON file.
        logger: Logger for logging the export status.
    """
    geo_df.to_file(file_path, driver="GeoJSON")
    if os.path.exists(file_path):
        logger.info(f"{file_name} exported successfully.")
    else:
        logger.error(f"Error exporting {file_name}.")

def plot_geodataframe(geodf, title, output_path, logger, filter_states=('AK', 'HI'),
                      crs="EPSG:4326", basemap_source=ctx.providers.OpenStreetMap.Mapnik):
    """
    Plots a GeoDataFrame, applies filters, sets CRS, and saves the plot.

    Parameters:
    - geodf: GeoDataFrame to plot.
    - title: Title of the plot.
    - output_path: File path to save the plot.
    - logger: Logger for logging actions and results.
    - filter_states: Tuple of state codes to exclude from the plot.
    - crs: Coordinate reference system to set for the GeoDataFrame.
    - basemap_source: Contextily basemap source for the background map.
    """
    try:
        # Filter to exclude specified states
        gdf_filtered = geodf[~geodf['state'].isin(filter_states)]

        # Check and set CRS to EPSG:4326 if necessary
        if gdf_filtered.crs is None or gdf_filtered.crs.to_string() != crs:
            gdf_filtered = gdf_filtered.set_crs(crs)

        # Convert to Web Mercator (EPSG:3857) for basemap compatibility
        gdf_filtered = gdf_filtered.to_crs(epsg=3857)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf_filtered.plot(
            ax=ax,
            markersize=40,      # Adjust point size as needed
            color='green',      # Set a bright color for the points
            edgecolor='black',  # Outline color for contrast
            linewidth=0.5,      # Outline width
            alpha=0.8           # Slight transparency
        )

        # Add a basemap from OpenStreetMap
        ctx.add_basemap(ax, source=basemap_source)
        ax.set_title(title)
        ax.set_axis_off()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        logger.info(f"Plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")

def process_parks_data(create_plots=True):
    logger = setup_logger()
    logger.info("Starting park transformation process.")

    # Define URLs and output directory
    csv_url = "https://raw.githubusercontent.com/pineapple-bois/USNationalParks/main/DATA/Masters/parks.csv"
    api_url = "https://api.github.com/repos/pineapple-bois/USNationalParks/contents/DATA/nps_boundary.geojson"
    output_dir = "FinalData"
    os.makedirs(output_dir, exist_ok=True)

    # Load and clean data
    print("Loading parks CSV data...")
    df = pd.read_csv(csv_url)
    logger.info("Loaded parks CSV data successfully.")

    print("Loading parks GeoJSON data...")
    gdf = fetch_geojson(api_url, logger)

    # Match and update parks
    name_matches = match_parks(df, gdf, logger)
    updated_gdf = update_geojson(gdf, name_matches, logger)

    # Add missing parks if necessary
    keywords = ['Pinnacles']
    pattern = '|'.join(keywords)
    matching_gdf = gdf[gdf['UNIT_NAME'].str.contains(pattern, case=False, na=False)]
    geo_df = gpd.GeoDataFrame(pd.concat([updated_gdf, matching_gdf]), crs="EPSG:4326")

    # Ensure data consistency
    print("Ensuring data consistency between CSV and GeoJSON...")
    ensure_data_consistency(df, geo_df, logger)

    # Prepare DataFrame for merging
    prepared_df = prepare_parks_dataframe(df, logger)

    # Create and export GeoJSON for park points
    geo_df_points, prepared_df = create_points_geojson(prepared_df, output_dir, logger)
    geojson_path = os.path.join(output_dir, "parks_shapes.geojson")

    # Create and export GeoJSON for park shapes
    geo_df_shapes = merge_shape_data(prepared_df, geo_df, logger)
    export_geojson(geo_df_shapes, geojson_path, "parks_shapes.geojson", logger)

    if create_plots:
        # Plot park points
        plot_geodataframe(geo_df_points, title='National Parks of Contiguous USA',
                      output_path="../Images/USParksLatLong.png", logger=logger)

        # Plot park shapes
        plot_geodataframe(geo_df_shapes, title='National Parks of Contiguous USA',
                      output_path="../Images/USParksShapes.png", logger=logger)

    print("Park transformation process completed.")
    logger.info("Park transformation process completed.")

    return geo_df_points, geo_df_shapes
