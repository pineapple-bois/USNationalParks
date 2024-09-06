import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import re
import os
from fuzzywuzzy import process
import contextily as ctx
from shapely.geometry import Point

## master url for GeoJSON data
# https://catalog.data.gov/dataset/national-park-boundaries/resource/cee04cfe-f439-4a65-91c0-ca2199fa5f93

def plot_geodataframe(geodf, title, output_path, filter_states=('AK', 'HI'),
                      crs="EPSG:4326", basemap_source=ctx.providers.OpenStreetMap.Mapnik):
    """
    Plots a GeoDataFrame, applies filters, sets CRS, and saves the plot.

    Parameters:
    - gdf: GeoDataFrame to plot
    - title: Title of the plot
    - output_path: File path to save the plot
    - filter_states: Tuple of state codes to exclude from the plot
    - crs: Coordinate reference system to set for the GeoDataFrame
    - basemap_source: Contextily basemap source for the background map
    """
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

    # Set title and hide axes for better visualization
    ax.set_title(title)
    ax.set_axis_off()

    # Save plot to file
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to {output_path}")

# Function to clean park names by removing "National Park", "National Parks", and specific variations
def clean_park_name(name):
    return re.sub(r' National Parks?$| National Park and Preserve$', '', name, flags=re.IGNORECASE).strip()

# convert acres to square kilometers
def acres_to_sq_km(acres):
    return round(acres * 0.00404686, 2)

csv_url = "https://raw.githubusercontent.com/pineapple-bois/USNationalParks/main/DATA/Masters/parks.csv"
df = pd.read_csv(csv_url)

api_url = "https://api.github.com/repos/pineapple-bois/USNationalParks/contents/DATA/nps_boundary.geojson"

# Fetch the download URL for the GeoJSON file from the GitHub API
try:
    response = requests.get(api_url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    file_info = response.json()
    download_url = file_info['download_url']
    gdf = gpd.read_file(download_url)
    assert gdf.shape == (510, 10), "The GeoDataFrame is not the required dimension"
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"An error occurred: {err}")

df_filter = df.copy()
df_filter['Cleaned_Park_Name'] = df_filter['Park Name'].apply(clean_park_name)

# Search for each cleaned park name in the GeoJSON
name_matches = {}
for index, row in df_filter.iterrows():
    park_code = row['Park Code']
    cleaned_name = row['Cleaned_Park_Name']

    # Try to match the cleaned park name in UNIT_NAME of GeoJSON
    matching_rows = gdf[gdf['UNIT_NAME'].str.contains(cleaned_name, case=False, na=False)]

    if not matching_rows.empty:
        matched_name = matching_rows['UNIT_NAME'].values[0]
        matched_code = matching_rows['UNIT_CODE'].values[0]
        name_matches[cleaned_name] = (matched_name, matched_code, park_code) 
    else:
        print(f"No match found for {cleaned_name} in GeoJSON")

# Identify unmatched parks by cleaned name
unmatched_csv = [row['Cleaned_Park_Name'] for _, row in df_filter.iterrows() if row['Cleaned_Park_Name'] not in name_matches]

# Granular searching for unmatched parks using fuzzy matching
for park_name in unmatched_csv:
    best_match = process.extractOne(park_name, gdf['UNIT_NAME'], scorer=process.fuzz.partial_ratio)
    if best_match and best_match[1] > 80:  
        matched_name = best_match[0]
        matched_code = gdf.loc[gdf['UNIT_NAME'] == matched_name, 'UNIT_CODE'].values[0]
        csv_code = df_filter.loc[df_filter['Cleaned_Park_Name'] == park_name, 'Park Code'].values[0]
        name_matches[park_name] = (matched_name, matched_code, csv_code)
        print(f"Fuzzy matched {park_name} (CSV) with {matched_name} (GeoJSON) - Match score: {best_match[1]}")
    else:
        print(f"No good fuzzy match found for {park_name} in GeoJSON")

# Create a mapping dictionary of GeoJSON UNIT_CODE to CSV park_code
mapping_dict = {matched_code: csv_code for _, (matched_name, matched_code, csv_code) in name_matches.items()}

# Update GeoJSON with the correct park codes, filtering only for 'National Park'
for matched_code, csv_code in mapping_dict.items():
    gdf.loc[(gdf['UNIT_CODE'] == matched_code) & (gdf['UNIT_TYPE'] == 'National Park'), 'UNIT_CODE'] = csv_code

# Extract the geometry objects for the updated park codes where UNIT_TYPE is 'National Park'
updated_gdf = gdf[(gdf['UNIT_CODE'].isin(mapping_dict.values())) & (gdf['UNIT_TYPE'] == 'National Park')]

assert updated_gdf.shape == (55, 10), "The GeoDataFrame is not the required dimension"

# Extract sets of park codes
park_set_df = set(df_filter["Park Code"])
park_set_gdf = set(updated_gdf['UNIT_CODE'])

# Find mismatches
geo_diff = park_set_df.difference(park_set_gdf)
csv_diff = park_set_gdf.difference(park_set_df)

# Define the keywords to match in the UNIT_NAME column
keywords = ['Pinnacles']
pattern = '|'.join(keywords) 

# Filter the GeoDataFrame for rows where UNIT_NAME contains any of the keywords
matching_gdf = gdf[gdf['UNIT_NAME'].str.contains(pattern, case=False, na=False)]

geo_df = pd.concat([updated_gdf, matching_gdf])
assert geo_df.shape[0] == 56, "The DataFrame has incorrect dimension"

park_set_df = set(df_filter["Park Code"])
park_set_gdf = set(geo_df['UNIT_CODE'])

try:
    assert park_set_df == park_set_gdf, (f"Park codes do not match: Missing in GeoDataFrame: {park_set_df - park_set_gdf}, "
                                         f"Extra in GeoDataFrame: {park_set_gdf - park_set_df}")
except AssertionError as e:
    print(e)
    raise
else:
    print("All park codes match correctly between CSV and GeoDataFrame.")

geo_df = geo_df[['UNIT_CODE', 'geometry']]

## Cleaning parks.csv
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df['square_km'] = df['acres'].apply(acres_to_sq_km)
df['park_name'] = df['park_name'].str.replace(r' National Park.*', '', case=False, regex=True)
df_transform = df.copy()

# Create POINT geometry from latitude and longitude
df_transform['geometry'] = df_transform.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
df_transform = df_transform.drop(columns=['acres', 'latitude', 'longitude'])
geo_df_points = gpd.GeoDataFrame(df_transform, geometry='geometry', crs="EPSG:4326")
if geo_df_points.crs != "EPSG:4326":
    geo_df_points = geo_df_points.to_crs("EPSG:4326")

assert isinstance(geo_df_points, gpd.GeoDataFrame), "The GeoDataFrame is not a GeoPandas object as required"
assert geo_df_points.shape == (56, 5), "The GeoDataFrame is not the required dimension"
assert not geo_df_points.isna().any().any(), "There are NaN values in geo_points DataFrame"

# Export points GeoDataFrame
geo_df_points.to_file("DATA/parks_points.geojson", driver="GeoJSON")
if os.path.exists("DATA/parks_points.geojson"):
    print("parks_points.geojson exported successfully")
else:
    print("Error exporting parks_points.geojson")

# Plot park points
plot_geodataframe(geo_df_points, title='National Parks of Contiguous USA', output_path="Images/USParksLatLong.png")

## Creating a POLYGON geometry GeoJSON file `parks_shapes.geojson`
merged_parks = pd.merge(df, geo_df, left_on='park_code', right_on='UNIT_CODE')
merged_parks = merged_parks.drop(columns=['acres', 'latitude', 'longitude', 'UNIT_CODE'])
merged_geo_parks = gpd.GeoDataFrame(merged_parks, geometry=merged_parks['geometry'], crs="EPSG:4326")
if merged_geo_parks.crs != "EPSG:4326":
    merged_geo_parks = merged_geo_parks.to_crs("EPSG:4326")

assert isinstance(merged_geo_parks, gpd.GeoDataFrame), "The GeoDataFrame is not a GeoPandas object as required"
assert merged_geo_parks.shape == (56, 5), "The GeoDataFrame is not the required dimension"
assert not merged_geo_parks.isna().any().any(), "There are NaN values in geo_shape DataFrame"

# Export shapes GeoDataFrame
merged_geo_parks.to_file("DATA/parks_shapes.geojson", driver="GeoJSON")
if os.path.exists("DATA/parks_shapes.geojson"):
    print("parks_shapes.geojson exported successfully")
else:
    print("Error exporting parks_shapes.geojson")

# Plot park points
plot_geodataframe(merged_geo_parks, title='National Parks of Contiguous USA', output_path="Images/USParksShapes.png")
