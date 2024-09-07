import pandas as pd
import numpy as np
import requests
import re
import os
import logging
from collections import Counter, defaultdict
from fuzzywuzzy import process, fuzz
from itertools import islice
from Functions.transformation_functions import process_scientific_names, standardize_common_names, standardize_common_names_subspecies

# Configure logging
logging.basicConfig(
    filename='../transformation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Create a logger object
logger = logging.getLogger()

# Get .csv file url
api_url = "https://api.github.com/repos/pineapple-bois/USNationalParks/contents/DATA/Masters/species.csv"
# Send request to GitHub API to fetch file
try:
    response = requests.get(api_url)
    response.raise_for_status()
    file_info = response.json()
    download_url = file_info['download_url']
    df = pd.read_csv(download_url, low_memory=False)
    logger.info(f"Species data loaded successfully with shape: {df.shape}")
except requests.exceptions.HTTPError as http_err:
    logger.info(f"HTTP error occurred: {http_err}")
except Exception as err:
    logger.info(f"An error occurred: {err}")

df.columns = [col.lower().replace(" ", "_") for col in df.columns]
df.rename(columns={'unnamed:_13': 'unnamed'}, inplace=True)
filtered_df = df[df['unnamed'].notna()]

# Remove all text beyond " National Park" including variations like " and Preserve"
df['park_name'] = df['park_name'].str.replace(r' National Park.*', '', case=False, regex=True)

# Splitting species_id column at the hyphen to create a new `park_code` public key
df['park_code'] = df['species_id'].str.split('-').str[0]
cols = list(df.columns)  # Get the list of current columns
cols.insert(1, cols.pop(cols.index('park_code')))  # Move 'park_code' to the second position
df = df[cols]  # Reorder DataFrame

## Isolating Birds
birds = df[df.category == 'Bird']
assert birds.shape == (14601, 15), "The DataFrame is not of the required dimension"

## `Resident` and `Breeder` belong in `seasonality`.
keywords = ['Breeder', 'Resident']
pattern = '|'.join(keywords)
matching_df = birds[birds['conservation_status'].str.contains(pattern, case=False, na=False)]

# Count unique values where 'abundance' is 'Native'
unique_native_abundance = birds[birds['abundance'] == 'Native']['species_id'].nunique()
assert unique_native_abundance == 8, "Expected 8 values from abundance filter"

# Count unique values where 'nativeness' is 'Present'
unique_present_nativeness = birds[birds['nativeness'] == 'Present']['species_id'].nunique()
assert unique_present_nativeness == 8, "Expected 8 values from nativeness filter"


# Remove common_name values in `record_status` and shift all columns to the right, one column left
indices_to_shift = matching_df.index
start_pos = matching_df.columns.get_loc('record_status')
shifted_df = matching_df.loc[indices_to_shift].apply(
    lambda row: pd.Series(np.append(row[start_pos + 1:].values, pd.NA), index=row[start_pos:].index), axis=1
)

# Combine the unchanged part of the rows with the shifted part
matching_df.loc[indices_to_shift, matching_df.columns[start_pos:]] = shifted_df

# Apply the same changes to the parent and child DataFrames using indices
df.loc[indices_to_shift] = matching_df.loc[indices_to_shift]
birds.loc[indices_to_shift] = matching_df.loc[indices_to_shift]
birds = birds.drop(columns=['unnamed'])

## `seasonality` column: Fill NaN values with 'Unknown'
birds['seasonality'] = birds['seasonality'].fillna('Unknown')
priority_keywords = ['Winter', 'Summer', 'Breeder', 'Migratory', 'Resident', 'Vagrant']

def simplify_seasonality(value):
    for keyword in priority_keywords:
        if keyword in value:
            return keyword
    return 'Unknown'

birds['seasonality'] = birds['seasonality'].apply(simplify_seasonality)

## Dealing with `NaN` values and conversion to categorical columns
## `conservation_status` as per IUCN Red List: https://en.wikipedia.org/wiki/IUCN_Red_List
# Fill NaN with `Least Concern` meaning 'non-protected' where all other values mean 'protected'

fill_values = {
    'conservation_status': 'Least Concern',
    'abundance': 'Unknown',
    'nativeness': 'Unknown',
    'occurrence': 'Not Confirmed'
}

birds['conservation_status'] = birds['conservation_status'].fillna(fill_values['conservation_status'])
birds['abundance'] = birds['abundance'].fillna(fill_values['abundance'])
birds['nativeness'] = birds['nativeness'].fillna(fill_values['nativeness'])
birds['occurrence'] = birds['occurrence'].fillna(fill_values['occurrence'])

conservation_status_order = ['Least Concern', 'Species of Concern', 'In Recovery', 'Under Review', 'Threatened', 'Proposed Endangered', 'Endangered']
abundance_order = ['Rare', 'Uncommon', 'Unknown', 'Occasional', 'Common', 'Abundant']
nativeness_order = ['Not Native', 'Unknown', 'Native']
record_status_order = ['In Review', 'Approved']
occurrence_order = ['Not Present (False Report)', 'Not Present (Historical Report)', 'Not Present', 'Not Confirmed', 'Present']

# Covert columns to categorical
birds['record_status'] = pd.Categorical(birds['record_status'], categories=record_status_order, ordered=True)
birds['occurrence'] = pd.Categorical(birds['occurrence'], categories=occurrence_order, ordered=True)
birds['nativeness'] = pd.Categorical(birds['nativeness'], categories=nativeness_order, ordered=True)
birds['abundance'] = pd.Categorical(birds['abundance'], categories=abundance_order, ordered=True)
birds['conservation_status'] = pd.Categorical(birds['conservation_status'], categories=conservation_status_order, ordered=True)

# add boolean column 'is_protected'
birds['is_protected'] = birds.conservation_status != 'Least Concern'
birds = birds.drop(columns=['category'])

# Define expected data types for categorical columns
expected_categorical_types = {
    'record_status': pd.CategoricalDtype(categories=record_status_order, ordered=True),
    'occurrence': pd.CategoricalDtype(categories=occurrence_order, ordered=True),
    'nativeness': pd.CategoricalDtype(categories=nativeness_order, ordered=True),
    'abundance': pd.CategoricalDtype(categories=abundance_order, ordered=True),
    'conservation_status': pd.CategoricalDtype(categories=conservation_status_order, ordered=True),
}

for column, expected_dtype in expected_categorical_types.items():
    assert birds[column].dtype == expected_dtype, f"Column '{column}' does not have the expected categorical dtype."

assert birds['is_protected'].dtype == 'bool', "The 'is_protected' column is not of boolean type."

for column, fill_value in fill_values.items():
    assert birds[column].isna().sum() == 0, f"Column '{column}' still contains NaN values."
    assert (birds[column] == fill_value).sum() > 0, f"Column '{column}' does not have any instances of the fill value '{fill_value}'."


## Classifying Species Records

# Searching for punctuation in scientific name
punctuation_pattern = r"[^\w\s,]"
punctuation_matches = birds[birds['scientific_name'].str.contains(punctuation_pattern, na=False)]

def strip_punctuation(text):
    return re.sub(punctuation_pattern, ' ', text)

birds['scientific_name'] = birds['scientific_name'].apply(strip_punctuation)
assert birds.scientific_name.isna().sum() == 0, "Unexpected NaN values under scientific_name"
assert birds.common_names.isna().sum() == 280, f"{birds.common_names.isna().sum()} NaN values expected under common_names"

sci_name_set = set(birds.scientific_name)

# Separate scientific names into standard and extended based on word count
single_sci_names = {name for name in sci_name_set if len(name.split()) == 1}
standard_sci_names = {name for name in sci_name_set if len(name.split()) == 2}
extended_sci_names = {name for name in sci_name_set if len(name.split()) > 2}

results = process_scientific_names(birds, condition=1, print_info=False)
single_sci_name_records = birds[birds['scientific_name'].isin(single_sci_names)]

# We drop the single name records for several reasons:
# 1. **Generic Common Names:** The common names listed are very generic and often refer to groups or types rather than specific species, which can lead to ambiguity and confusion.
# 2. **Review Status:** The fact that all but one of these records are marked as “In Review” suggests that these records are not finalized and are potentially under investigation or pending confirmation.
# 3. **Data Quality and Relevance:** The goal is to create a well-defined OLAP database with precise species information.

# Save the single scientific name records to a CSV for backup
single_sci_name_records.to_csv('DATA/Backups/single_sci_name_birds.csv', index=True)

# Drop the records with single scientific names from the birds DataFrame in place
birds.drop(birds[birds['scientific_name'].isin(single_sci_names)].index, inplace=True)

sci_name_set = set(birds.scientific_name)
single_sci_names = {name for name in sci_name_set if len(name.split()) == 1}
assert single_sci_names == set(), "single_sci_names is not an empty set as required"


## Standard `scientific_name`
results2 = process_scientific_names(birds, condition=2, print_info=False)
no_common_names = results2['no_common_names']
multiple_common_names = results2['multiple_common_names']
single_common_names = results2['single_common_names']


# Extract all records from birds where scientific_name is in no_common_names
no_common_names_records = birds[birds['scientific_name'].isin(no_common_names)]
assert no_common_names_records.shape == (41, 14), "no_common_names DataFrame is not the required dimension"

# Save the no_common_names_records to a CSV for backup
no_common_names_records.to_csv('DATA/Backups/no_common_names_birds.csv', index=True)

approved_records = no_common_names_records[no_common_names_records['record_status'] == 'Approved']
approved_genera = approved_records['scientific_name'].apply(lambda x: x.split()[0]).unique()
genus_dataframes = {genus: birds[birds['scientific_name'].str.startswith(genus)] for genus in approved_genera}

# Accessing each DataFrame by genus name
glaucidium_df = genus_dataframes.get('Glaucidium')
geothlypis_df = genus_dataframes.get('Geothlypis')
eromophila_df = genus_dataframes.get('Eromophila')

all_sci_names = birds['scientific_name'].unique()
potential_matches = {}

for sci_name in no_common_names:
    # Use fuzzy matching to find the best matches in the list of all scientific names
    matches = process.extract(sci_name, all_sci_names, scorer=fuzz.ratio, limit=5)  # Adjust limit as needed
    
    # Filter matches to include only those with a similarity score over 90
    high_quality_matches = [match for match in matches if match[1] > 90]
    if len(high_quality_matches) > 1:
        # Retrieve common names for the matching scientific names
        common_names = birds[birds['scientific_name'].isin([match[0] for match in high_quality_matches])]['common_names'].dropna()

        common_names_counter = Counter([name.strip() for names in common_names for name in names.split(',')])
        most_common_name = common_names_counter.most_common(1)
        
        potential_matches[sci_name] = {
            'matches': high_quality_matches,
            'most_common_name': most_common_name[0][0] if most_common_name else 'No common name found'
        }

# Corrected scientific names mapping based on investigation
corrected_names = {
    'Lophortyx californica': 'Lophortyx californicus',  
    'Eromophila alpestris': 'Eremophila alpestris',
    'Peucaea cassini': 'Peucaea cassinii', 
    'Peucaea casinii': 'Peucaea cassinii',
    'Geothlypis tolomiei': 'Geothlypis tolmiei',
    'Glaucidium californicum': 'Glaucidium gnoma'
}

# Corresponding common names for the corrected scientific names
common_name_updates = {
    'Lophortyx californicus': 'California Quail',
    'Eremophila alpestris': 'Horned Lark',
    'Peucaea cassinii': "Cassin's Sparrow",
    'Geothlypis tolmiei': "Macgillivray's Warbler",
    'Glaucidium gnoma': 'Mountain Pygmy Owl, Northern Pygmy-Owl'
}

for old_name, new_name in corrected_names.items():
    birds.loc[birds['scientific_name'] == old_name, 'scientific_name'] = new_name

for sci_name, common_name in common_name_updates.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = common_name

birds_updated = birds[birds['scientific_name'].isin(corrected_names.values())]

results2 = process_scientific_names(birds, condition=2, print_info=False)
no_common_names = results2['no_common_names']
multiple_common_names = results2['multiple_common_names']
single_common_names = results2['single_common_names']
birds.drop(birds[birds['scientific_name'].isin(no_common_names)].index, inplace=True)


## Multiple `common_names`

# Apply the function to the multiple_common_names
standardized_name_mapping = standardize_common_names(multiple_common_names)
assert isinstance(standardized_name_mapping, dict) and len(standardized_name_mapping) == 369, \
    (f"Expected standardized_name_mapping to be a dictionary of length 369. "
     f"Got length {len(standardized_name_mapping)} and type {type(standardized_name_mapping)}")


# Extract the first elements (scientific names) from the list of tuples
multi_sci_names_list = [sci_name for sci_name, counts in multiple_common_names]
multi_common_names_records = birds[birds['scientific_name'].isin(multi_sci_names_list)]
multi_common_names_records.to_csv('DATA/Backups/multi_common_names_birds.csv', index=True)

# Update the birds DataFrame with the standardized common names
for sci_name, common_name in standardized_name_mapping.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = common_name


## Extended `scientific_name`

results3 = process_scientific_names(birds, condition=3, print_info=False)
no_common_names = results3['no_common_names']
multiple_common_names = results3['multiple_common_names']
single_common_names = results3['single_common_names']

no_common_names_records = birds[birds['scientific_name'].isin(no_common_names)]
no_common_names_records.to_csv('DATA/Backups/no_common_names_subspecies_birds.csv', index=True)


## Handling Subspecies in Scientific Names: https://en.wikipedia.org/wiki/Subspecies
# A subspecies is a taxonomic classification below species, representing populations of a species that are genetically distinct due to geographic or ecological factors.
# Subspecies names are often included in scientific naming as a third part, following the genus and species (e.g., *Buteo jamaicensis borealis*).
# **Matching on Genus and Species**: We extracted the genus and species parts of the scientific names and used fuzzy matching to find existing records with associated common names.
# **Updating Common Names**: For each match, the most appropriate common name was selected and updated to include the subspecies designation in parentheses (e.g., *Red-Tailed Hawk (borealis subspecies)*).

all_sci_names = birds['scientific_name'].unique()
potential_matches = {}

for sci_name in no_common_names:
    genus_species = ' '.join(sci_name.split()[:2])

    # Use fuzzy matching to find best matches in the list of all scientific names
    matches = process.extract(genus_species, all_sci_names, scorer=fuzz.ratio, limit=5)
    high_quality_matches = [match for match in matches if match[1] > 90]

    if high_quality_matches:
        # Retrieve common names for the matching scientific names
        matched_sci_names = [match[0] for match in high_quality_matches]
        common_names = birds[birds['scientific_name'].isin(matched_sci_names)]['common_names'].dropna()

        # Count occurrences of each common name
        common_names_counter = Counter([name.strip() for names in common_names for name in names.split(',')])
        most_common_name = common_names_counter.most_common(1)

        # Store potential matches and the most common name found
        potential_matches[sci_name] = {
            'matches': high_quality_matches,
            'most_common_name': most_common_name[0][0] if most_common_name else 'No common name found'
        }

updated_common_names = {}
for sci_name, info in potential_matches.items():
    most_common_name = info['most_common_name']
    subspecies = sci_name.split()[2]  
    updated_name = f"{most_common_name} ({subspecies} subspecies)"
    updated_common_names[sci_name] = updated_name

# Update the birds DataFrame with the new common names
for sci_name, common_name in updated_common_names.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = common_name


results3 = process_scientific_names(birds, condition=3, print_info=False)
no_common_names = results3['no_common_names']
no_common_names_records = birds[birds['scientific_name'].isin(no_common_names)]

# Dropping the remaining unnamed subspecies
birds.drop(birds[birds['scientific_name'].isin(no_common_names)].index, inplace=True)


## Multiple `common_names`
standardized_names_mapping = standardize_common_names_subspecies(multiple_common_names)
multi_sci_names_list = [sci_name for sci_name, counts in multiple_common_names]

# Return a subset of the birds DataFrame where scientific_name matches any in the list
multi_common_names_records = birds[birds['scientific_name'].isin(multi_sci_names_list)]
assert multi_common_names_records.shape == (123, 14), "multi_common_names_records is not the required dimension"

# Update the birds DataFrame with the standardized common names
for sci_name, common_name in standardized_names_mapping.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = common_name


results3 = process_scientific_names(birds, condition=3, print_info=False)
single_common_names = results3['single_common_names']

## Remaining NaN records in `birds`

nan_common = birds[birds.common_names.isna()]
assert nan_common.shape == (41, 14),  "nan_common is not the required dimension"
nan_sci_names = nan_common.scientific_name.unique().tolist()

all_sci_names = birds['scientific_name'].unique()
potential_matches = {}

for sci_name in nan_sci_names:
    genus_species = ' '.join(sci_name.split()[:2])

    # Use fuzzy matching to find best matches in the list of all scientific names
    matches = process.extract(genus_species, all_sci_names, scorer=fuzz.ratio, limit=5)
    high_quality_matches = [match for match in matches if match[1] > 90]

    if high_quality_matches:
        # Retrieve common names for the matching scientific names
        matched_sci_names = [match[0] for match in high_quality_matches]
        common_names = birds[birds['scientific_name'].isin(matched_sci_names)]['common_names'].dropna()

        # Count occurrences of each common name
        common_names_counter = Counter([name.strip() for names in common_names for name in names.split(',')])
        most_common_name = common_names_counter.most_common(1)

        # Store potential matches and the most common name found
        potential_matches[sci_name] = {
            'matches': high_quality_matches,
            'most_common_name': most_common_name[0][0] if most_common_name else 'No common name found'
        }

updated_common_names = {}
for sci_name, info in potential_matches.items():
    most_common_name = info['most_common_name']
    sci_name_parts = sci_name.split()
    if len(sci_name_parts) > 2:
        subspecies = sci_name_parts[2]  # Get the subspecies part
        updated_name = f"{most_common_name} ({subspecies} subspecies)"
    else:
        updated_name = most_common_name

    updated_common_names[sci_name] = updated_name

for sci_name, common_name in updated_common_names.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = common_name

# Check if there are still any NaN values in 'common_names'
assert birds.common_names.isna().sum() == 0, "birds.common_names NaN count is not zero as required"


## Single `common_names`

# Searching for 'unusual' records
MAX_LENGTH = 50  # Define a threshold for very long common names
MAX_WORD_COUNT = 6  # Define a word count threshold
UNUSUAL_PUNCTUATION_PATTERN = r"[^a-zA-Z\s,\'\-\(\)/`´]"  # Allows letters, spaces, commas, apostrophes, and hyphens

long_names = birds[birds['common_names'].str.len() > MAX_LENGTH]
high_word_count = birds[birds['common_names'].str.split().apply(len) > MAX_WORD_COUNT]
unusual_punctuation = birds[birds['common_names'].str.contains(UNUSUAL_PUNCTUATION_PATTERN, na=False)]

unusual_common_names = pd.concat([long_names, high_word_count, unusual_punctuation]).drop_duplicates()

# Change manually for genus species records and numeric/strange characters
birds.loc[111, 'common_names'] = 'Black Guillemot'
birds.loc[26297, 'common_names'] = "Yellow Warbler"
birds.loc[26777, 'common_names'] = "Pileated Woodpecker"
birds.loc[19387, 'common_names'] = 'Northern Three-Toed Woodpecker'
birds.loc[42377, 'common_names'] = "Lucy's Warbler"

# Drop the record with index 30773 - absolute mess
birds.drop(index=30773, inplace=True)
assert not 30773 in birds.index, "30773 exists in birds df"

## Cross-referencing potential typos with the common name counts will provide a data-driven way to determine which scientific name is likely correct.
# We do not have domain expertise in taxonomic nomenclature or ornithology...

sci_name_common_name_counts = {}
for sci_name in birds['scientific_name'].unique():
    common_names = birds[birds['scientific_name'] == sci_name]['common_names']
    all_common_names = [name.strip() for names in common_names.dropna() for name in names.split(',')]
    sci_name_common_name_counts[sci_name] = Counter(all_common_names)


birds_sorted = birds.sort_values(by='scientific_name', ascending=True)
two_word_sci_names = birds_sorted[birds_sorted['scientific_name'].str.split().str.len() == 2]
three_or_more_word_sci_names = birds_sorted[birds_sorted['scientific_name'].str.split().str.len() >= 3]


## Reviewing `scientific_name` for multiple comma-separated entries under `common_names`

# Isolate records with comma-separated common names
comma_separated_common_names = two_word_sci_names[two_word_sci_names['common_names'].str.contains(',', na=False)]
subset_sci_names = comma_separated_common_names['scientific_name'].unique()
potential_matches = {}

for sci_name in subset_sci_names:
    # Extract genus and species from the scientific name
    genus_species = ' '.join(sci_name.split()[:2])

    # Use fuzzy matching to find best matches in the list of all scientific names
    matches = process.extract(genus_species, all_sci_names, scorer=fuzz.ratio, limit=5)
    high_quality_matches = [match for match in matches if match[1] > 90]

    if high_quality_matches:
        # Retrieve common names for the matching scientific names
        matched_sci_names = [match[0] for match in high_quality_matches]
        common_names = birds[birds['scientific_name'].isin(matched_sci_names)]['common_names'].dropna()

        # Count occurrences of each full common name as a whole string
        common_names_counter = Counter(common_names)

        # Only consider potential matches with multiple distinct common names
        if len(common_names_counter) > 1:
            most_common_name = common_names_counter.most_common(1)

            # Store potential matches and the most common name found, along with the counter
            potential_matches[sci_name] = {
                'matches': high_quality_matches,
                'most_common_name': most_common_name[0][0] if most_common_name else 'No common name found',
                'common_name_counts': common_names_counter  # Include the Counter object with counts of full strings
            }

# Create mappings for scientific names and common names based on potential matches
scientific_name_mapping = {}
common_name_mapping = {}

for sci_name, info in potential_matches.items():
    best_match_sci_name = info['matches'][1][0]  
    best_common_name = info['most_common_name']
    scientific_name_mapping[sci_name] = best_match_sci_name
    common_name_mapping[sci_name] = best_common_name

# Update the common_names
birds['common_names'] = birds.apply(
    lambda row: common_name_mapping.get(row['scientific_name'], row['common_names']), axis=1
)

# Update the scientific_name
birds['scientific_name'] = birds.apply(
    lambda row: scientific_name_mapping.get(row['scientific_name'], row['scientific_name']), axis=1
)


## Looking for typos/ambiguities under matching common names where `scientific_name` is of form: *Genus species*

common_name_to_sci_names = {}
for common_name in two_word_sci_names['common_names'].unique():
    associated_sci_names = two_word_sci_names[two_word_sci_names['common_names'] == common_name]['scientific_name']
    sci_name_counts = Counter(associated_sci_names)
    common_name_to_sci_names[common_name] = sci_name_counts

potential_issues = {}
for common_name, sci_name_counts in common_name_to_sci_names.items():
    if len(sci_name_counts) > 1:
        potential_issues[common_name] = sci_name_counts


# Convert potential issues to a DataFrame
potential_issues_df = pd.DataFrame([
    {'common_names': common_name, 
     'scientific_name': sci_name, 
     'record_count': count
     } 
    for common_name, sci_name_counts in potential_issues.items()
    for sci_name, count in sci_name_counts.items()
])

assert potential_issues_df.shape == (265, 3), "potential_issues_df is not the required dimension"

# Create a dictionary to hold indices and affected records for each common name
affected_records = {}
issues_common_names = set(potential_issues_df.common_names)

for common_name in issues_common_names:
    records = birds[(birds['common_names'] == common_name) & 
                    (birds['scientific_name'].str.split().apply(len) == 2)]
    indices = records.index.tolist()
    if indices:  
        affected_records[common_name] = records

# Combine all affected records into a single DataFrame for exporting
affected_records_df = pd.concat(affected_records.values())
assert affected_records_df.shape == (3270, 14), "affected_records_df is not the required dimension"
assert set(potential_issues_df.common_names) == set(affected_records_df.common_names), "The sets are not equal as required"
assert all(affected_records_df['scientific_name'].str.split().apply(len) == 2), "There are subspecies or incorrect scientific names present"

# Save the affected records to a CSV file for backup before making changes
affected_records_df.to_csv('DATA/Backups/multi_sci_names_birds.csv', index=True)


# Create a mapping of common names to the selected scientific name based on highest count, and flag ties for manual review
selected_sci_names = {}
ties_for_review = {}
for common_name, group in potential_issues_df.groupby('common_names'):
    max_count = group['record_count'].max()
    max_count_names = group[group['record_count'] == max_count]
    
    if len(max_count_names) > 1:
        # Flag for manual review due to tie
        ties_for_review[common_name] = max_count_names[['scientific_name', 'record_count']].to_dict(orient='records')
    else:
        chosen_name = max_count_names.iloc[0]['scientific_name']
        selected_sci_names[common_name] = chosen_name

# The above records, tied on count were reviewed manually on wikipedia
manual_choices = {
    "Abert's Towhee": "Melozone aberti",
    "Black-Footed Albatross": "Phoebastria nigripes",
    "Far Eastern Curlew": "Numenius madagascariensis",
    "Gray-Headed Chickadee": "Poecile cinctus",
    "Lawrence's Warbler": "Vermivora lawrencei",
    "Wandering Tattler": "Tringa incana"
}

selected_sci_names.update(manual_choices)
assert len(selected_sci_names) == 126, f"Expecting 126 amendment, recieved {len(selected_sci_names)}"


# Create a mapping of common names to selected scientific names
selected_sci_names_mapping = {common_name: sci_name for common_name, sci_name in selected_sci_names.items()}

birds['scientific_name'] = birds.apply(
    lambda row: selected_sci_names_mapping.get(row['common_names'], row['scientific_name']) 
    if len(row['scientific_name'].split()) == 2 else row['scientific_name'], 
    axis=1
)

two_word_sci_names = birds_sorted[birds_sorted['scientific_name'].str.split().str.len() == 2]
sci_name_to_common_names = defaultdict(list)

for index, row in two_word_sci_names.iterrows():
    sci_name = row['scientific_name']
    common_name = row['common_names']
    sci_name_to_common_names[sci_name].append(common_name)

sci_name_to_common_names = {sci_name: list(set(common_names)) for sci_name, common_names in sci_name_to_common_names.items()}
assert len(sci_name_to_common_names) == 961, "sci_name_to_common_names is not the required length"

comma_separated_names = {
    sci_name: list(set(common_names)) 
    for sci_name, common_names in sci_name_to_common_names.items() 
    if any(',' in name for name in common_names)
}


## Check for Hawaiian birds based on the presence of an acute accent character in common names
special_char_birds = two_word_sci_names[two_word_sci_names['common_names'].str.contains('´', na=False)]

correction_mappings = {
    '´': 'ʻ',  # Replace acute accent with okina
    '_': ' ',
    'L': 'l'
}

def correct_hawaiian_names(name):
    for incorrect, correct in correction_mappings.items():
        name = name.replace(incorrect, correct)
    return name

birds['common_names'] = birds.apply(
    lambda row: correct_hawaiian_names(row['common_names']) if row['scientific_name'] in special_char_birds['scientific_name'].values else row['common_names'], 
    axis=1
)

# Display the corrected DataFrame
corrected_hawaiian_birds = birds[birds['scientific_name'].isin(special_char_birds['scientific_name'])]
assert corrected_hawaiian_birds.shape == (20, 14), "corrected_hawaiian_birds is not the required dimension"


## Reviewing subspecies of form *Genus species subspecies*

birds_sorted = birds.sort_values(by='scientific_name', ascending=True)
two_word_sci_names = birds_sorted[birds_sorted['scientific_name'].str.split().str.len() == 2].copy()
three_word_sci_names = birds_sorted[birds_sorted['scientific_name'].str.split().str.len() >= 3].copy()


# Group by genus and species to identify subspecies variations
three_word_sci_names.loc[:, 'genus_species'] = three_word_sci_names['scientific_name'].apply(lambda x: ' '.join(x.split()[:2]))

# Create a mapping of genus_species to common names from two_word_sci_names
genus_species_to_common_name = two_word_sci_names.set_index('scientific_name')['common_names'].to_dict()
three_word_sci_names.loc[:, 'matched_common_name'] = three_word_sci_names['genus_species'].map(genus_species_to_common_name)

# Exclude specific columns from three_word_sci_names 
columns_to_exclude = ['record_status', 'occurrence', 'nativeness', 'abundance', 'seasonality', 'conservation_status', 'is_protected'] 
three_word_sci_names_filtered = three_word_sci_names.drop(columns=columns_to_exclude, errors='ignore') 

assert three_word_sci_names_filtered.shape == (526, 9), "three_word_sci_names_filtered is not the required dimension"
three_word_sci_names_filtered.head()

# list of common names to exclude to inspection
common_names_to_exclude = ["Mountain Pygmy Owl, Northern Pygmy-Owl", 
                           "Baltimore Oriole, Northern Oriole", 
                           "Western Gull",
                           "Whimbrel",
                           "Lazuli Bunting",
                           "American Three-Toed Woodpecker",
                           "Snail Kite",
                           "Great Gray Owl",
                           "Spotted Owl",
                           "Prairie Chicken",
                           "Golden-Winged Warbler",
                           "Blue-Winged Warbler"
                           ]  

# Filter out rows where 'matched_common_name' is in the list of values to exclude
three_word_sci_names = three_word_sci_names[~three_word_sci_names['matched_common_name'].isin(common_names_to_exclude)]

columns_to_exclude = ['record_status', 'occurrence', 'nativeness', 'abundance', 'seasonality', 'conservation_status', 'is_protected'] 
three_word_sci_names = three_word_sci_names.drop(columns=columns_to_exclude, errors='ignore')
three_word_sci_names_filtered = three_word_sci_names.dropna(subset=['matched_common_name'])

# Update the 'common_names' column to include subspecies information if not already present
def standardize_common_names(row):
    subspecies = row['scientific_name'].split()[-1]
    
    # Check if the 'common_names' already contains brackets
    if '(' not in row['common_names']:
        new_common_name = f"{row['matched_common_name']} ({subspecies} subspecies)"
        return new_common_name
    else:
        # If brackets already present, keep the original common name
        return row['common_names']

three_word_sci_names_filtered.loc[:, 'common_names'] = three_word_sci_names_filtered.apply(standardize_common_names, axis=1)
assert three_word_sci_names_filtered.shape == (468, 9), "three_word_sci_names_filtered is not the required dimension"

filtered_indices = three_word_sci_names_filtered.index
birds_filtered_subset = birds.loc[filtered_indices]
assert birds_filtered_subset.shape == (468, 14), "birds_filtered_subset is not the required dimension"

# Save the subset to a CSV file as backup
birds_filtered_subset.to_csv('DATA/Backups/subspecies_sci_name_birds.csv', index=False)

# Update the 'common_names' in the original birds DataFrame using the indices from three_word_sci_names_filtered
birds.loc[filtered_indices, 'common_names'] = three_word_sci_names_filtered['common_names']


## Identify `common_names` with multiple `scientific_name`:

common_name_sci_count = birds.groupby('common_names')['scientific_name'].nunique()
multi_sci_common_names = common_name_sci_count[common_name_sci_count > 1]

# Log these results
logger.info("Common names with multiple associated scientific names:")
logger.info(multi_sci_common_names)

multi_sci_common_names_list = multi_sci_common_names.index.tolist()
multi_sci_common_name_records = birds[birds['common_names'].isin(multi_sci_common_names_list)]

common_name_to_sci_names_counts = {}
for common_name, group in multi_sci_common_name_records.groupby('common_names'):
    sci_names_counts = Counter(group['scientific_name'])
    common_name_to_sci_names_counts[common_name] = sci_names_counts

# Define the updates for common names with the subspecies information in parentheses
updates = {
    'Picoides dorsalis fasciatus': "American Three-Toed Woodpecker (fasciatus subspecies)",
    'Junco hyemalis shufeldti': "Dark-Eyed Junco (shufeldti subspecies)",
    'Junco hyemalis thurberi': "Dark-Eyed Junco (thurberi subspecies)",
    'Strix nebulosa nebulosa': "Great Gray Owl (nebulosa subspecies)",
    'Falco peregrinum anatum': "Peregrine Falcon (anatum subspecies)"
}

for sci_name, new_common_name in updates.items():
    birds.loc[birds['scientific_name'] == sci_name, 'common_names'] = new_common_name

# Filter records with NaN in the 'family' column
nan_family_records = birds[birds['family'].isna()]
known_families = birds.dropna(subset=['family']).set_index(['order', 'scientific_name'])['family'].to_dict()

# Attempt to fill missing 'family' values by matching 'order' and 'scientific_name'
for index, row in nan_family_records.iterrows():
    order = row['order']
    scientific_name = row['scientific_name']
    family = known_families.get((order, scientific_name))
    if family:
        birds.at[index, 'family'] = family

assert birds['family'].isna().sum() == 0, f"NaN records present under 'family': {birds['family'].isna().sum()} records missing"


## Isolating Birds of Prey: We create a list of birds of prey 'groups' to search under `common_name`

birds_of_prey = ["Eagle", "Hawk", "Falcon", "Buzzard", "Harrier", "Kite", "Owl", "Osprey", 
                 "Vulture", "Condor", "Kestrel", 'Buteo', 'Accipiter', 'Caracara']

pattern = '|'.join(birds_of_prey)

def find_raptors(common_names):
    # Convert to string and handle NaN values
    if pd.isna(common_names):
        return ''
    # Find keywords that are present in the common_names
    matches = set()  # Use a set to avoid duplicates
    for keyword in birds_of_prey:
        if keyword in common_names:
            matches.add(keyword)
    return ', '.join(matches)

birds['raptor_group'] = birds['common_names'].apply(find_raptors)

total_raptors_comm = birds['raptor_group'].loc[birds['raptor_group'] != ''].count()
assert total_raptors_comm == 1341, f"total_raptors_comm is of length {total_raptors_comm}. Expected length 1341"


## Birds of Prey Scientific Families and Genera

# - Accipitridae (Hawks, Eagles, and relatives)
# - Falconidae (Falcons)
# - Harpagiidae (Harriers)
# - Pandionidae (Ospreys)
# - Accipitridae (Kites)
# - Cathartidae (New World Vultures)
# - Buteo (Buzzards and Buteos)
# - Accipiter (Goshawks and Accipiters)
# - Tytonidae (Barn Owls)
# - Strigidae (Typical Owls)

# *Caveat emptor*: This list may not be comprehensive.

birds_of_prey_sci = [
    "Accipitridae", "Falconidae", "Harpagiidae", "Pandionidae", "Cathartidae", "Buteo", "Accipiter",
    "Tytonidae", "Strigidae"
]

pattern = '|'.join(birds_of_prey_sci)
birds['raptor_sci_fam'] = birds['family'].str.findall(f'({pattern})')
birds['raptor_sci_fam'] = birds['raptor_sci_fam'].apply(lambda x: ', '.join(x) if isinstance(x, list) and x else '')

raptors_df = (birds[(birds.raptor_group != '') | (birds.raptor_sci_fam != '')])
assert raptors_df.shape == (1470, 16), "raptors_df is not the required dimension"

raptors_df = raptors_df.copy()
mask = raptors_df['raptor_group'] == ''
raptors_df.loc[mask, 'ambiguous'] = True
result = raptors_df[raptors_df['ambiguous'] == True]

# *Accipiter gentilis* is `Northern Goshawk`, a type of `Hawk`
# *Falco columbarius* is `Merlin`, a type of `Falcon`

raptors_df = raptors_df.copy()
merlin_gyrfalcon_mask = raptors_df['common_names'].str.contains(r'Merlin|Gyrfalcon', case=False, regex=True)
northern_goshawk_mask = raptors_df['common_names'].str.contains(r'Northern Goshawk', case=False, regex=True)

raptors_df.loc[merlin_gyrfalcon_mask, 'raptor_common'] = 'Falcon'
raptors_df.loc[northern_goshawk_mask, 'raptor_common'] = 'Hawk'

result = raptors_df[raptors_df['ambiguous'] == True]
assert result.shape == (129, 18), "result is not the required dimension"

# Extract indices from the result DataFrame
updated_indices = result.index
birds.loc[updated_indices, 'raptor_group'] = result['raptor_common']
assert not birds.loc[updated_indices, 'raptor_group'].isna().any(), "NaN values present in 'raptor_common' for the specified indices"

hawk_owl_indices = birds[birds['raptor_group'] == 'Hawk, Owl'].index
birds.loc[hawk_owl_indices, 'raptor_group'] = "Owl"
assert birds[birds['raptor_group'] == 'Hawk, Owl'].empty, "There are records with 'raptor_group' set to 'Hawk, Owl'."

# Change the empty string in 'raptor_group' to "N/A"
birds['raptor_group'] = birds['raptor_group'].replace('', 'N/A')

# Create a new boolean column 'is_raptor' based on 'raptor_group'
birds['is_raptor'] = birds['raptor_group'].apply(lambda x: x != 'N/A')
assert birds['is_raptor'].dtype == bool, "is_raptor is not a boolean column"

# Reorder the columns to place 'raptor_group' next to 'common_names'
columns_order = birds.columns.tolist()  # Get the current order of columns
common_names_index = columns_order.index('common_names')
columns_order.insert(common_names_index + 1, columns_order.pop(columns_order.index('raptor_group')))
birds = birds[columns_order] 

# Export the 'birds' DataFrame to a CSV file and pickle file
csv_path = '../DATA/birds.csv'
pickle_path = '../DATA/birds.pkl'

try:
    # Attempt to export the DataFrame to CSV
    birds.to_csv(csv_path, index=False)

    # Attempt to export the DataFrame to a pickle file
    birds.to_pickle(pickle_path)

    # Check if both files were created successfully
    if os.path.exists(csv_path) and os.path.exists(pickle_path):
        logger.info(f"Data successfully exported to {csv_path} and {pickle_path}")
    else:
        raise Exception("Export failed: One or both files were not created successfully.")

except Exception as e:
    logger.info(f"An error occurred during export: {e}")
