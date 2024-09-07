import pandas as pd
import geopandas as gpd
import numpy as np
import random
import logging
import os

# Configure logging
logging.basicConfig(
    filename='../transformation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Read in pickles data for datatype logging
birds = pd.read_pickle('../DATA/birds.pkl')

logger.info(f"Shape of birds: {birds.shape}")
logger.info(f"Data Types:\n{birds.dtypes}")
logger.info(f"Value Count:\n{birds.nunique()}")

nan_counts_per_column = birds.isna().sum()
logger.info(f"NaN counts:\n{nan_counts_per_column}")


birds = birds.rename(columns={'common_names': 'common_name'})
birds = birds.reset_index(drop=True)
birds.head()

# We create a subset DataFrame of species information only
species = birds[['order', 'family', 'scientific_name', 'common_name', 'raptor_group']]
species = species.drop_duplicates()
species = species.sort_values(by='scientific_name')

# Group by scientific name and count occurrences
duplicates = species.groupby('scientific_name').size()
duplicates = duplicates[duplicates > 1]

logger.info(f"Duplicated scientific names:\n{duplicates}")

# List of duplicated scientific names
duplicated_sci_names = ['Phylloscopus borealis', 'Polioptila caerulea']
duplicated_records = species[species['scientific_name'].isin(duplicated_sci_names)]

# Correct families for the scientific names
correct_families = {
    'Phylloscopus borealis': 'Phylloscopidae', 
    'Polioptila caerulea': 'Polioptilidae'   
}

# Update the family in the birds DataFrame
for sci_name, correct_family in correct_families.items():
    birds.loc[birds['scientific_name'] == sci_name, 'family'] = correct_family

# Update the family in the species DataFrame
for sci_name, correct_family in correct_families.items():
    species.loc[species['scientific_name'] == sci_name, 'family'] = correct_family

species = species.drop_duplicates()
species = species.sort_values(by='scientific_name')
species = species.reset_index(drop=True)
assert species.shape[0] == 1177, "The records do not have the correct dimension"


# Create a species_id column with leading zeros (e.g., 0001, 0002, ...)
species['species_code'] = species.reset_index().index + 1  # Start with 1
species['species_code'] = species['species_code'].apply(lambda x: f"{x:04d}")  # Format with leading zeros for 4 digits
species = species[['species_code'] + [col for col in species.columns if col != 'species_code']]


## Matching species data with the park records

birds_transform = birds.copy()
birds_transform = birds_transform.merge(species[['species_code', 'order', 'family', 'scientific_name', 'common_name', 'raptor_group']],
                    on=['order', 'family', 'scientific_name', 'common_name', 'raptor_group'], 
                    how='left')

columns_to_drop = ['species_id', 'park_name', 'order', 'family', 'scientific_name', 'common_name', 'raptor_group']
birds_transform = birds_transform.drop(columns=columns_to_drop)

cols = birds_transform.columns.tolist()
cols.insert(cols.index('park_code') + 1, cols.pop(cols.index('species_code')))  # Move species_code after park_code
birds_transform = birds_transform[cols]

# Log unique values of specific categorical columns in their defined order
columns_to_log = ['occurrence', 'nativeness', 'abundance', 'seasonality', 'conservation_status']

for col in columns_to_log:
    if isinstance(birds[col].dtype, pd.CategoricalDtype):
        ordered_categories = birds[col].cat.categories
        value_counts = birds[col].value_counts().reindex(ordered_categories, fill_value=0)  # Fill missing categories with zero counts
        logger.info(f"Unique values for '{col}'\nDefined order:\n{value_counts}\n")
    else:
        # Fallback: log unique counts if the column is not categorical
        unique_counts = birds[col].value_counts(dropna=False)  # Include NaNs in the count if present
        logger.info(f"Unique values for '{col}':\n{unique_counts}\n")

# Exporting birds_transform and species DataFrames to CSV
try:
    # Export the transformed DataFrames
    birds_transform.to_csv('DATA/records.csv', index=False)
    species.to_csv('DATA/species.csv', index=False)

    # Confirm successful exports
    if os.path.exists('../DATA/records.csv') and os.path.exists('../DATA/species.csv'):
        logger.info("birds_transform.csv and species.csv exported successfully.")
    else:
        raise Exception("Export failed: One or both files were not created successfully.")

except Exception as e:
    logger.error(f"An error occurred while exporting birds_transform and species DataFrames: {e}")


## Randomly selecting 10 indices from `birds` DataFrame to recreate the records in a test environment
random.seed(42)
selected_indices = random.sample(range(birds.shape[0]), 10)
selected_records = birds.iloc[selected_indices]

# Create a dictionary with indices and the required data
selected_data_dict = {
    idx: {
        'order': row['order'],
        'family': row['family'],
        'scientific_name': row['scientific_name'],
        'common_name': row['common_name'],
        'raptor_group': row['raptor_group']
    } for idx, row in selected_records.iterrows()
}
logger.info("Random Indices for testing:")
for i, data in selected_data_dict.items():
    logger.info(f"{i}: {data}")
