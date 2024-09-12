import pandas as pd
import yaml
from collections import Counter

from ExtractTransform.utils import DataFrameUtils, DataFrameTransformation
from ExtractTransform.transform_strategies.strategy_factory import TransformStrategyFactory


# Add to below as required as and when abstract base methods written and testing completed
VALID_CATEGORIES = ['Mammal', 'Bird', 'Reptile']


class TransformSpecies:
    """
    TransformSpecies class subsets the transformed master data by category and applies specific transformations.

    Category options include:
        - 'Mammal'
        - 'Bird'
        - 'Reptile'
        - 'Amphibian'
        - 'Fish'
        - 'Vascular Plant'
        - 'Spider/Scorpion'
        - 'Insect'
        - 'Invertebrate'
        - 'Fungi'
        - 'Nonvascular Plant'
        - 'Crab/Lobster/Shrimp'
        - 'Slug/Snail'
        - 'Algae'

    Attributes:
        category (str): The category of species to filter from the DataFrame.
        dataframe (pd.DataFrame): The subset of the DataFrame containing only the specified category.
        logger (logging.Logger): A logger object configured to log information specific to the category.
        records_dropped (int): Counter for the number of records dropped during transformations.

    Methods:
        __init__(category: str, dataframe: pd.DataFrame = None, pickle_path: str = None):
            Initializes the TransformSpecies class, sets up logging for the specified category,
            and loads the species data either from a provided DataFrame or from a pickle file.
            It verifies the integrity of the DataFrame, subsets the data by the specified category,
            and applies category-specific transformations.

        _subset_category() -> pd.DataFrame:
            Filters the transformed DataFrame to include only the specified category.
    """
    def __init__(self, category, dataframe=None, pickle_path=None):
        if not isinstance(category, str):
            raise TypeError("Category must be a string.")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Valid options are: {', '.join(VALID_CATEGORIES)}")

        self.category = category
        formatted_category = self.category.lower().replace(" ", "_").replace("/", "_")
        self.logger = DataFrameUtils.setup_logger(f'transform_{formatted_category}',
                                                  f'transform_{formatted_category}.log')
        self.records_dropped = 0

        # Load the DataFrame from the provided pickle path or accept the given DataFrame
        if dataframe is not None:
            self.dataframe = dataframe
            self.verify_dataframe(self.dataframe)
        elif pickle_path:
            self.logger.info(f"Loading DataFrame from {dataframe_path}")
            try:
                self.dataframe = pd.read_pickle(dataframe_path)
                self.verify_dataframe(self.dataframe)  # Verify the DataFrame
            except Exception as e:
                self.logger.error(f"Error loading DataFrame: {e}")
                raise
        else:
            raise ValueError("A valid dataframe or dataframe path must be provided to load species data.")

        try:
            self.dataframe = self._subset_category()  # Subset by category
        except Exception as e:
            self.logger.error(f"Error subsetting data for category '{self.category}': {e}")
            raise

        try:
            self._apply_category_transformations()
        except Exception as e:
            self.logger.error(f"Error applying transformations for category '{self.category}': {e}")
            raise

        print(f"{self.category} data created:\n{self.dataframe.species_id.nunique()} records"
              f"\n{self.dataframe.species_id.nunique()} unique scientific names")
        self.logger.info(f"DataFrame created for category '{self.category}' with shape: {self.dataframe.shape}")
        self.logger.info(f"Records dropped during creation: {self.records_dropped}")
        DataFrameUtils.pickle_data(self.dataframe,
                                   f'FinalData/Species',
                                   f'{formatted_category}.pkl',
                                   self.logger
                                   )

    def verify_dataframe(self, dataframe):
        """
        Verifies that the DataFrame meets the required structure and content for species data.

        Checks include:
        - Presence of essential columns.
        - Correct data types for critical columns.
        - Validating the shape or row count if necessary.

        Args:
            dataframe (pd.DataFrame): The DataFrame to verify.

        Raises:
            ValueError: If any verification check fails.
        """
        expected_columns = {
            'species_id': 'object',
            'park_code': 'object',
            'scientific_name': 'object',
            'common_names': 'object',
            'category': 'object',
            'record_status': 'category',
            'occurrence': 'category',
            'nativeness': 'category',
            'abundance': 'category',
            'seasonality': 'object',
            'conservation_status': 'category',
            'is_protected': 'bool'
        }

        # Check for missing columns
        missing_columns = set(expected_columns.keys()) - set(dataframe.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

        # Check data types
        for column, expected_type in expected_columns.items():
            actual_type = str(dataframe[column].dtype)
            if actual_type != expected_type:
                self.logger.error(f"Column '{column}' expected type '{expected_type}', but got '{actual_type}'.")
                raise ValueError(f"Column '{column}' expected type '{expected_type}', but got '{actual_type}'.")

        # Check the shape or number of rows
        expected_shape = (119247, 15)  # Adjust based on your data's expected shape
        if dataframe.shape != expected_shape:
            self.logger.warning(f"DataFrame shape {dataframe.shape} differs from expected {expected_shape}.")
            # You can decide if this should raise an error or just log a warning

        self.logger.info("DataFrame verification passed:\n")

    def _subset_category(self):
        """Filters the transformed DataFrame to include only the specified category."""
        return self.dataframe[self.dataframe['category'] == self.category]

    def _remove_genus_only_records(self):
        """Identifies and removes records that contain only the genus name in the 'scientific_name' field,
        logs the actions, and saves these records to a CSV file.
        """
        df_copy = self.dataframe.copy()
        # Extract records with only genus in scientific name
        single_name_records = {name for name in set(df_copy['scientific_name']) if len(name.split()) == 1}
        df_to_save = df_copy[df_copy['scientific_name'].isin(single_name_records)].copy(deep=True)

        # Log and save the records with genus-only names
        self.logger.info(f"Found {len(df_to_save)} Genus-only (generic) records.")
        DataFrameUtils.save_dataframe_to_csv(df_to_save, f"BackupData/{self.category}",
                                             "genus_records_dropped.csv", self.logger)
        self.records_dropped += len(df_to_save)
        self.logger.info(f"{len(df_to_save)} records dropped.\n")

        df_copy.drop(df_to_save.index, inplace=True)
        self.dataframe = df_copy

        return self.dataframe

    def _process_no_common_names(self, condition):
        """
        Processes scientific names with no associated common names based on the given condition.
        Updates the dataframe with new common names.
        """
        no_common_names_records, no_common_names = DataFrameTransformation.identify_records(
            self.dataframe, condition, 'no_common_names', self.logger, self.category
        )

        original_count = len(self.dataframe)
        # Fuzzy match and update common names
        self.dataframe = DataFrameTransformation.fuzzy_match_and_update(
            self.dataframe, no_common_names, 'common_names', self.logger, self.category, identifier=condition
        )
        self.records_dropped += original_count - len(self.dataframe)  # Update records dropped counter

        return self.dataframe

    def _process_multiple_common_names(self, condition):
        """
        Processes scientific names with multiple associated common names based on the given condition.
        Updates the dataframe with standardized common names.
        """
        multiple_common_names_records, multiple_common_names = DataFrameTransformation.identify_records(
            self.dataframe, condition, 'multiple_common_names', self.logger, self.category
        )

        original_count = len(self.dataframe)
        # Standardize the common names for identified records
        standardize_method = (
            DataFrameTransformation.standardize_common_names_subspecies
            if condition == 3 else DataFrameTransformation.standardize_common_names
        )
        self.dataframe = DataFrameTransformation.standardize_names(
            self.dataframe, multiple_common_names, standardize_method, self.logger, self.category, condition
        )
        self.records_dropped += original_count - len(self.dataframe)  # Update records dropped counter

        return self.dataframe

    def _process_subspecies(self):
        """
        Handles subspecies records by processing scientific names with no common names and
        those with multiple common names specifically for subspecies.
        """
        # Process subspecies with no common names
        self.dataframe = self._process_no_common_names(condition=3)
        # Process subspecies with multiple common names
        self.dataframe = self._process_multiple_common_names(condition=3)

        return self.dataframe

    def _update_records_nan_common_names(self):
        """
        Identifies records with NaN in 'common_names' and updates them using fuzzy matching.
        """
        nan_common_names_records = self.dataframe[self.dataframe['common_names'].isna()]
        nan_sci_names = nan_common_names_records['scientific_name'].unique().tolist()
        self.logger.info(f"Found {nan_common_names_records.shape[0]} scientific names with no associated common name.")
        DataFrameUtils.save_dataframe_to_csv(nan_common_names_records, f"BackupData/{self.category}",
                                             "nan_common_names.csv", self.logger)

        # Perform fuzzy matching to find potential common names
        matches = DataFrameTransformation.fuzzy_match_scientific_names(nan_sci_names, self.dataframe)
        updated_common_names = set()
        for sci_name, info in matches.items():
            most_common_name = info['most_common_name']
            if most_common_name == 'No common name found':
                self.logger.info(f"Skipping update for {sci_name} as no valid common name was found.")
                continue

            # Update only the 'common_names' field for the matched scientific name
            self.dataframe.loc[self.dataframe['scientific_name'] == sci_name, 'common_names'] = most_common_name
            updated_common_names.add(sci_name)
        self.logger.info(f"Matched and updated {len(updated_common_names)} scientific names.")

        records_to_drop = self.dataframe[
            self.dataframe['scientific_name'].isin(nan_sci_names) & ~self.dataframe['scientific_name'].isin(updated_common_names)
            ]
        self.logger.info(f"Dropping {len(records_to_drop)} records that were not updated.\n")
        self.records_dropped += len(records_to_drop)
        self.dataframe = self.dataframe.drop(records_to_drop.index)

        return self.dataframe

    def _update_comma_sep_common_names(self):
        """
        Updates common names by resolving ambiguities based on category-specific logic, with special handling
        for Birds and Mammals using manually defined mappings from a configuration file.
        """
        # Process potential matches but do not create automatic mappings
        sci_name_set = self.dataframe['scientific_name'].unique()
        potential_matches = DataFrameTransformation.process_comma_separated_names(
            self.dataframe, sci_name_set, match_score=90
        )

        # Save potential matches to YAML for review
        readable_potential_matches = {
            sci_name: {
                'matches': [{'match': match[0], 'score': match[1]} for match in info['matches']],
                'most_common_name': info['most_common_name'],
                'common_name_counts': dict(info['common_name_counts'])
            }
            for sci_name, info in potential_matches.items()
        }

        DataFrameUtils.save_dict_to_yaml(
            readable_potential_matches,
            f'BackupData/{self.category}/Reviews',
            'process_comma_separated_names.yaml',
            self.logger
        )
        self.logger.warning(f"Found {len(potential_matches)} matches requiring manual review.")

        # Load manual updates from YAML for Birds and Mammals
        common_name_mapping = {}
        scientific_name_mapping = {}
        # Load manual updates from YAML for Birds and Mammals
        if self.category in ['Bird', 'Mammal']:
            manual_choices = DataFrameUtils.load_dict_from_yaml("config/update_common_names.yaml")
            category_choices = manual_choices.get("manual_choices", {}).get(self.category, {})

            # Update mappings and log them immediately after
            scientific_name_mapping.update(category_choices.get('updates', {}).get('scientific_name_updates', {}))
            common_name_mapping.update(category_choices.get('updates', {}).get('common_name_updates', {}))

            self.logger.info(f"Scientific Name Mapping after update: {scientific_name_mapping}")
            self.logger.info(f"Common Name Mapping after update: {common_name_mapping}")

        # Unified helper function to update rows based on custom mappings
        def update_row(row, common_name_mapping, scientific_name_mapping, updated_indices):
            sci_name = row['scientific_name']
            new_sci_name = scientific_name_mapping.get(sci_name, sci_name)
            new_common_name = common_name_mapping.get(new_sci_name, row['common_names'])
            if new_sci_name != sci_name or new_common_name != row['common_names']:
                updated_indices.add(row.name)
                return new_common_name, new_sci_name
            return row['common_names'], row['scientific_name']

        # Apply updates using the helper function for Birds and Mammals
        updated_indices = set()
        if self.category in ['Bird', 'Mammal']:
            self.logger.info(f"Applying custom mappings for category '{self.category}'.")
            self.dataframe[['common_names', 'scientific_name']] = self.dataframe.apply(
                lambda row: pd.Series(update_row(row, common_name_mapping, scientific_name_mapping, updated_indices)),
                axis=1
            )

            self.logger.info(f"Updated {len(updated_indices)} rows.\n")
        else:
            self.logger.warning("No custom mappings applied. Verify selections in process_comma_separated_names.yaml.")

        return self.dataframe


    def _resolve_sci_name_ambiguities(self):
        """
        Resolves common name ambiguities by identifying potential typos or ambiguities in common names
        linked to 'Genus species' scientific names, and updates the DataFrame accordingly.
        This method is designed to be generic but can include category-specific mappings or exceptions.
        """
        potential_issues_df = DataFrameTransformation.identify_sci_name_ambiguities(
                            self.dataframe, self.logger, self.category
        )
        selected_sci_names, ties_for_review = DataFrameTransformation.select_scientific_names_by_common_name(
            potential_issues_df)
        self.logger.info(f"Selected scientific names for {len(selected_sci_names)} common names based on highest counts.")
        DataFrameUtils.save_dict_to_yaml(selected_sci_names,
                                         f'BackupData/{self.category}/Reviews',
                                         'resolve_sci_name_ambiguities_selected.yaml',
                                         self.logger)

        if ties_for_review:
            self.logger.warning(f"Found {len(ties_for_review)} common names with ties requiring manual review.")
            # Save to YAML for review
            DataFrameUtils.save_dict_to_yaml(ties_for_review,
                                             f'BackupData/{self.category}/Reviews',
                                             'resolve_sci_name_ambiguities_ties.yaml',
                                             self.logger)

        # Apply manual choices from configuration
        manual_choices = DataFrameUtils.load_dict_from_yaml("config/manual_choices.yaml")
        category_manual_choices = manual_choices.get("manual_choices", {}).get(self.category, {})
        if not category_manual_choices:
            self.logger.info(
                f"No manual choices found for category '{self.category}'. Proceeding without applying manual updates.")
        else:
            selected_sci_names.update(category_manual_choices)

        # Update the scientific names in the DataFrame
        original_count = len(self.dataframe)
        self.dataframe = DataFrameTransformation.update_scientific_names(self.dataframe, selected_sci_names)
        updated_indices = self.dataframe.index[self.dataframe['scientific_name'].isin(selected_sci_names.values())]
        updated_records_df = self.dataframe.loc[updated_indices]
        updated_count = len(updated_records_df)

        if updated_count > 0:
            self.logger.info(f"Updated {updated_count} records with new scientific names.\n")
        else:
            self.logger.info("No records updated with new scientific names.\n")
        self.records_dropped += (original_count - len(self.dataframe))

        return self.dataframe

    def _standardize_subspecies_common_names(self):
        """
        Standardizes the common names for subspecies in the DataFrame by mapping them to their
        corresponding genus-species common names, appending subspecies information where applicable.
        """
        config = DataFrameUtils.load_dict_from_yaml('config/subspecies_config.yaml')

        subspecies_df = DataFrameTransformation.identify_subspecies(self.dataframe)
        subspecies_df = DataFrameTransformation.map_genus_species_to_common_names(self.dataframe, subspecies_df)
        updated_subspecies_df = DataFrameTransformation.filter_and_standardize_subspecies_names(
            subspecies_df, self.category, config, self.logger
        )

        filtered_indices = updated_subspecies_df.index
        self.dataframe.loc[filtered_indices, 'common_names'] = updated_subspecies_df['common_names']
        self.logger.info(f"Processed and updated {len(updated_subspecies_df)} subspecies common names.\n")

        return self.dataframe

    def _update_common_names_with_subspecies(self):
        """
        Identifies common names associated with multiple scientific names and updates
        them with subspecies information as specified in the configuration.
        """
        # Identify common names associated with multiple scientific names
        common_name_sci_count = self.dataframe.groupby('common_names')['scientific_name'].nunique()
        multi_sci_common_names = common_name_sci_count[common_name_sci_count > 1]
        multi_sci_common_names_list = multi_sci_common_names.index.tolist()
        multi_sci_common_name_records = self.dataframe[self.dataframe['common_names'].isin(multi_sci_common_names_list)]

        common_name_to_sci_names_counts = {}
        for common_name, group in multi_sci_common_name_records.groupby('common_names'):
            sci_names_counts = Counter(group['scientific_name'])
            common_name_to_sci_names_counts[common_name] = dict(sci_names_counts)

        DataFrameUtils.save_dict_to_yaml(common_name_to_sci_names_counts,
                                         f'BackupData/{self.category}/Reviews',
                                         'update_common_names_with_subspecies.yaml',
                                         self.logger)

        # Load updates from the configuration
        manual_choices = DataFrameUtils.load_dict_from_yaml('config/manual_verification.yaml')
        category_updates = manual_choices.get("manual_choices", {}).get(self.category, {}).get('updates', {})
        # Check if there are any updates for the current category
        if not category_updates:
            self.logger.info(f"No specific updates found for category '{self.category}'. Skipping manual updates.")
        else:
            for sci_name, new_common_name in category_updates.items():
                self.dataframe.loc[self.dataframe['scientific_name'] == sci_name, 'common_names'] = new_common_name
            self.logger.info(f"Applied {len(category_updates)} manual updates.")

        common_name_sci_count = self.dataframe.groupby('common_names')['scientific_name'].nunique()
        multi_sci_common_names = common_name_sci_count[common_name_sci_count > 1]
        self.logger.info(f"Common names with multiple associated scientific names:\n{multi_sci_common_names}\n")

        return self.dataframe

    def _update_missing_family(self):
        """
        Attempts to fill missing 'family' values by matching 'order' and 'scientific_name'.
        Saves the records with NaN in the 'family' column before and after the update.
        """
        # Filter records with NaN in the 'family' column
        nan_family_records = self.dataframe[self.dataframe['family'].isna()]
        self.logger.info(f"Found {len(nan_family_records)} records with missing 'family' values.")

        # Save the initial records with NaN family values for reference
        if not nan_family_records.empty:
            DataFrameUtils.save_dataframe_to_csv(
                nan_family_records,
                f"BackupData/{self.category}",
                "nan_family_records_before_update.csv",
                self.logger
            )

        # Create a dictionary of known families indexed by 'order' and 'scientific_name'
        known_families = self.dataframe.dropna(subset=['family']).set_index(['order', 'scientific_name'])[
            'family'].to_dict()

        # Attempt to fill missing 'family' values by matching 'order' and 'scientific_name'
        updated_count = 0
        for index, row in nan_family_records.iterrows():
            order = row['order']
            scientific_name = row['scientific_name']
            family = known_families.get((order, scientific_name))
            if family:
                self.dataframe.at[index, 'family'] = family
                updated_count += 1
        self.logger.info(f"Updated {updated_count} records with missing 'family' values.\n")

        return self.dataframe

    def verify_dataset_integrity(self):
        """
        Verifies the integrity of the dataset by checking for NaN values,
        dropping duplicates, and ensuring the count of unique scientific names
        matches the count of unique records. Logs the results of these checks.

        Returns:
            bool: True if integrity checks pass, otherwise raises an exception or
                  returns a DataFrame of discrepancies for manual update.
        """
        # Check for NaN values in the dataframe
        if self.dataframe.isna().any().any():
            nan_counts = self.dataframe.isna().sum()
            nan_fields = nan_counts[nan_counts > 0].index.tolist()
            self.logger.error(f"Integrity check failed: Found NaN values in fields: {nan_fields}")
            raise ValueError("Dataset contains NaN values. Please address missing data before proceeding.")

        # Drop duplicates and count unique records
        species = self.dataframe[['order', 'family', 'scientific_name', 'common_names']]
        species = species.drop_duplicates()
        unique_records_count = species.shape[0]
        unique_sci_names_count = species['scientific_name'].nunique()

        if unique_sci_names_count != unique_records_count:
            self.logger.error(
                f"Integrity check failed: Mismatch between unique scientific names ({unique_sci_names_count}) "
                f"and unique records ({unique_records_count})."
            )

            # Find discrepancies for manual review
            discrepancies = species.groupby('scientific_name').filter(lambda x: len(x) > 1)
            self.logger.info("Returning discrepancies for manual update.")
            return discrepancies

        # Log success if all checks pass
        self.logger.info("Dataset integrity verified: No NaN values and counts match.\n")
        return True  # Indicating the integrity check passed

    def _apply_category_transformations(self):
        """
        Applies the appropriate transformation strategy based on the category.

        This method performs a series of data transformations that are initially common
        to all categories. It then conditionally applies additional transformations specific
        to the 'Bird' category, including handling genus-species ambiguities, subspecies
        standardizations, and family field updates.

        For other categories, such as 'Mammal', this method currently skips these specific
        transformations but provides a framework for future expansion.

        Steps to scale this method for new categories:
        - Test each transformation step individually to ensure compatibility with the new category.
        - Add category-specific conditional branches similar to the 'Bird' category where needed.
        - Update the `TransformStrategyFactory` with new strategies as new categories are added.

        Returns:
            None: Updates are applied directly to the class instance's dataframe attribute.

        Example:
            To extend this method to handle 'Mammal' category, add:
                if self.category == 'Mammal':
                    # Insert Mammal-specific transformation methods here
        """

        # General transformations applicable to all categories
        self.dataframe = self._remove_genus_only_records()
        self.logger.info("Processing Records with no 'common_names':")
        self.dataframe = self._process_no_common_names(condition=2)
        self.logger.info("Processing Records with multiple 'common_names':")
        self.dataframe = self._process_multiple_common_names(condition=2)
        self.logger.info("Processing Records containing subspecies (Genus species subspecies):")
        self.dataframe = self._process_subspecies()
        self.logger.info("Processing Records where 'common_names' is NaN:")
        self.dataframe = self._update_records_nan_common_names()

        # Bird-specific transformations
        if self.category in ['Bird', 'Mammal']:
            self.logger.info("Remapping comma-separated 'common_names':")
            self.dataframe = self._update_comma_sep_common_names()

            self.logger.info("Resolving Genus species ambiguities:")
            self.dataframe = self._resolve_sci_name_ambiguities()

            self.logger.info("Resolving subspecies ambiguities:")
            self.dataframe = self._standardize_subspecies_common_names()

            self.logger.info("Resolving scientific_name ambiguities:")
            self.dataframe = self._update_common_names_with_subspecies()

            self.logger.info("Resolving missing family fields:")
            self.dataframe = self._update_missing_family()

            # Apply extra methods based on specific category strategy
            strategy = TransformStrategyFactory.get_strategy(self.category)
            self.dataframe = strategy.apply_transformations(self.dataframe, self.logger)

        # To scale to other categories, each method branched above should be tested
        # one by one with the new category to ensure compatibility and correctness.
        elif self.category == 'Reptile':
            self.logger.info("Remapping comma-separated 'common_names':")
            self.dataframe = self._update_comma_sep_common_names()

        else:
            # Define more categories as required
            pass
