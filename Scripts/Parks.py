# Merging Parks Data

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

df = pd.read_csv("../DATA/parks.csv")
gdf = gpd.read_file('../DATA/nps_boundary.geojson')
gdf_nat_parks = gdf[gdf['UNIT_TYPE'] == 'National Park']


park_set_df = set(df["Park Code"])
park_set_gdf = set(gdf_nat_parks['UNIT_CODE'])
geo_diff = park_set_df.difference(park_set_gdf)
csv_diff = park_set_gdf.difference(park_set_df)
matching_parks_gdf = gdf_nat_parks[gdf_nat_parks['UNIT_CODE'].isin(park_set_df)]


keywords = ['Pinnacles', 'Sequoia', 'Congaree']
pattern = '|'.join(keywords)  # This creates a pattern like 'Pinnacles|Sequoia|Congaree'

# Filter the GeoDataFrame for rows where UNIT_NAME contains any of the keywords
matching_gdf = gdf[gdf['UNIT_NAME'].str.contains(pattern, case=False, na=False)]

# Change the `UNIT_CODE` of Congaree and Sequoia to match the .csv and append to `matching_parks_gdf`

index_to_update = 415  
new_value = 'CONG'

if index_to_update in gdf.index:
    # Update the UNIT_CODE at the specific index
    matching_gdf.loc[index_to_update, 'UNIT_CODE'] = new_value
else:
    print(f"Index {index_to_update} does not exist in the DataFrame.")


index_to_update = 508  
new_value = 'SEKI'

if index_to_update in gdf.index:
    # Update the UNIT_CODE at the specific index
    matching_gdf.loc[index_to_update, 'UNIT_CODE'] = new_value
else:
    print(f"Index {index_to_update} does not exist in the DataFrame.")


geo_df = pd.concat([matching_parks_gdf, matching_gdf])
assert geo_df.shape[0] == 56, "The DataFrame has incorrect dimension"


park_set_df = set(df["Park Code"])
park_set_gdf = set(geo_df['UNIT_CODE'])
geo_diff = park_set_df.difference(park_set_gdf)

assert geo_diff == set(), "The difference is not an empty set as required"

geo_df = geo_df[['UNIT_CODE', 'geometry']]
assert geo_df.shape == (56, 2), "The DataFrame has incorrect dimensions"



# convert column to lowercase, replace spaces
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# convert acres to hectares
def acres_to_hectares(acres):
    return round(acres * 0.404686, 2)

df['hectares'] = df['acres'].apply(acres_to_hectares)

# Remove " National Park" from the park_name column
df['park_name'] = df['park_name'].str.replace(' National Park', '', case=False)

merged_parks = pd.merge(df, geo_df, left_on='park_code', right_on='UNIT_CODE')
merged_parks = merged_parks.drop(columns=['acres', 'latitude', 'longitude', 'UNIT_CODE'])

# Ensure the merged DataFrame is a GeoDataFrame with the correct geometry column
merged_geo_parks = gpd.GeoDataFrame(merged_parks, geometry=merged_parks['geometry'])

# Assert that there are no NaN values in merged_parks
assert not merged_geo_parks.isna().any().any(), "There are NaN values in merged_parks DataFrame"


# Check and set CRS to EPSG:4326 if necessary (WGS 84 - latitude/longitude)
if merged_geo_parks.crs is None or merged_geo_parks.crs.to_string() != 'EPSG:4326':
    merged_geo_parks = merged_geo_parks.to_crs(epsg=4326)

# Convert to Web Mercator (EPSG:3857) for basemap compatibility
merged_geo_parks = merged_geo_parks.to_crs(epsg=3857)

# Plot the GeoDataFrame
fig, ax = plt.subplots(figsize=(12, 10))
merged_geo_parks.plot(ax=ax, color='green', edgecolor='black', alpha=0.7)

# Add a basemap, from OpenStreetMap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set title and axis off for better visualization
ax.set_title('National Parks on Map of USA')
ax.set_axis_off()

# Show the plot
plt.show()




