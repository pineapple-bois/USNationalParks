import pandas as pd
import logging
import inspect
import random

from ExtractTransform.utils import DataFrameUtils


class TransformRecords:
    """
    TransformRecords class manages the transformation of species records.

    This class reads in DataFrames, creates a unique set of species, normalizes records,
    and handles any discrepancies in the data.
    """

    def __init__(self, dataframes):
        """
        Initializes the TransformRecords class with one or more DataFrames.

        Args:
            dataframes (list of pd.DataFrame): A list of DataFrames to be transformed.
        """
        if not isinstance(dataframes, list) or not all(isinstance(df, pd.DataFrame) for df in dataframes):
            raise ValueError("Input must be a list of DataFrames.")

        # Setup logger
        self.dataframes = dataframes
        self.logger = DataFrameUtils.setup_logger('transform_records', 'transform_records.log')
        self.logger.info(f"Initialized TransformRecords with {len(dataframes)} DataFrame(s).")
        self._assign_dataframes()
        self._finalize_records()
        self.verify_integrity_of_records()
        self.select_random_indices(count=15)

        # Process Birds
        bird_df = self.bird[['category', 'order', 'family', 'scientific_name',
                             'common_names', 'raptor_group', 'is_raptor']]
        self.bird_records, next_index = self.assign_species_codes(bird_df, 'order')
        bird_copy = self.bird_records.copy()
        bird_copy = bird_copy.drop(columns=['category'], errors='ignore')
        columns_order = ['species_code'] + [col for col in bird_copy.columns if col != 'species_code']
        bird_copy = bird_copy[columns_order]
        bird_copy = bird_copy.reset_index(drop=True)
        self.bird_records = bird_copy
        self.logger.info(f"Birds DataFrame has final shape: {self.bird_records.shape}")
        DataFrameUtils.pickle_data(self.bird_records,
                                   "Pipeline/FinalData",
                                   "bird_master.pkl",
                                   self.logger
                                   )

        # Process Mammals
        mammal_df = self.mammal[['category', 'order', 'family', 'scientific_name',
                                 'common_names', 'predator_group', 'is_large_predator']]
        self.mammal_records, next_index = self.assign_species_codes(mammal_df, 'order', start_index=next_index)
        mammal_copy = self.mammal_records.copy()
        mammal_copy = mammal_copy.drop(columns=['category'], errors='ignore')
        columns_order = ['species_code'] + [col for col in mammal_copy.columns if col != 'species_code']
        mammal_copy = mammal_copy[columns_order]
        mammal_copy = mammal_copy.reset_index(drop=True)
        self.mammal_records = mammal_copy
        self.logger.info(f"Mammals DataFrame has final shape: {self.mammal_records.shape}")
        DataFrameUtils.pickle_data(self.mammal_records,
                                   "Pipeline/FinalData",
                                   "mammal_master.pkl",
                                   self.logger
                                   )

        # Merge Species Code with Records
        species_df = pd.concat([self.bird_records, self.mammal_records], ignore_index=True)
        merge_keys = ['scientific_name']
        self.all_records = pd.merge(
                                self.records,
                                species_df[merge_keys + ['species_code']],
                                on=merge_keys,
                                how='left'
        )

        records_copy = self.all_records.copy()
        required_columns = [
            'park_code', 'species_code',
            'record_status', 'occurrence',
            'nativeness', 'abundance',
            'seasonality', 'conservation_status',
            'is_protected'
        ]
        records_copy = records_copy[required_columns]
        self.all_records = records_copy
        self.generate_data_dictionary(self.bird_records, self.mammal_records)

        # Remove duplicates
        duplicates = records_copy[records_copy.duplicated(subset=['park_code', 'species_code'], keep=False)]
        non_duplicates = records_copy.drop(duplicates.index)
        self.logger.info("Exporting duplicate records to CSV")
        DataFrameUtils.save_dataframe_to_csv(duplicates,
                                             "Pipeline/BackupData",
                                             "duplicate_records.csv",
                                             self.logger
                                             )
        deduplicated_duplicates = duplicates.groupby(['park_code', 'species_code']).apply(
                                                     TransformRecords.deduplicate_group).reset_index(drop=True)
        records_copy = pd.concat([non_duplicates, deduplicated_duplicates], ignore_index=True)
        records_copy = records_copy.reset_index(drop=True)
        dupes = records_copy[records_copy.duplicated(subset=['park_code', 'species_code'], keep=False)]
        assert dupes.shape[0] == 0, "Duplicate Records still exist"
        self.all_records = records_copy
        dupes_removed = len(self.all_records) - len(non_duplicates)
        self.logger.info(f"Removed {dupes_removed} duplicate records")
        self.logger.info(f"Records DataFrame has final shape: {self.all_records.shape}")
        DataFrameUtils.pickle_data(self.all_records,
                                   "Pipeline/FinalData",
                                   "record_master.pkl",
                                   self.logger
                                   )

    def _assign_dataframes(self):
        """
        Assigns DataFrames to specific class attributes based on their 'category' field.
        """
        mapping = {
            'Bird': 'bird',
            'Mammal': 'mammal',
            'Reptile': 'reptile'
        }

        for idx, df in enumerate(self.dataframes, start=1):
            if 'category' not in df.columns:
                self.logger.warning(f"DataFrame {idx} does not have a 'category' column. Skipping assignment.")
                continue

            unique_categories = df['category'].unique()
            if len(unique_categories) == 1:
                category = unique_categories[0]
                if category in mapping:
                    setattr(self, mapping[category], df)
                    self.logger.info(f"Assigned DataFrame {idx} to self.{mapping[category]}.")
            else:
                self.logger.warning(
                    f"DataFrame {idx} has multiple categories or missing 'category' values: {unique_categories}. Skipping assignment.")

    def _finalize_records(self):
        """
        Finalizes the records DataFrame by applying specific column dropping logic and
        concatenating the adjusted DataFrames.
        """
        # Apply column dropping logic first to ensure consistent dimensions
        bird_copy = self.bird.drop(columns=['raptor_group'], errors='ignore') if hasattr(self, 'bird') else pd.DataFrame()
        mammal_copy = self.mammal.drop(columns=['predator_group'], errors='ignore') if hasattr(self, 'mammal') else pd.DataFrame()

        # Use the reptile DataFrame as is
        reptile_copy = self.reptile.copy() if hasattr(self, 'reptile') else pd.DataFrame()

        # Concatenate the DataFrames now that they have consistent dimensions
        self.records = pd.concat([bird_copy, mammal_copy, reptile_copy],
                                 ignore_index=True).sort_values(by='species_id')
        self.records = self.records.reset_index(drop=True)
        self.logger.info(f"Initial records DataFrame created with shape: {self.records.shape}")

    def verify_integrity_of_records(self):
        """
        Verifies the integrity of the records by:
        - Copying the DataFrame and retaining only essential columns.
        - Dropping duplicates.
        - Checking for NaNs.
        - Ensuring the number of unique records matches the number of unique scientific names.

        Raises:
            ValueError: If any of the integrity checks fail.
        """
        records_copy = self.records.copy()
        essential_columns = ['category', 'order', 'family', 'scientific_name', 'common_names']
        records_copy = records_copy[essential_columns]
        records_copy = records_copy.drop_duplicates()

        if records_copy.isna().any().any():
            nan_columns = records_copy.columns[records_copy.isna().any()].tolist()
            raise ValueError(f"Integrity check failed: Found NaN values in columns: {nan_columns}")

        unique_records_count = len(records_copy)
        unique_sci_names_count = records_copy['scientific_name'].nunique()

        if unique_records_count != unique_sci_names_count:
            raise ValueError(
                f"Integrity check failed: Mismatch between unique records ({unique_records_count}) "
                f"and unique scientific names ({unique_sci_names_count})."
            )

        print("Data integrity check passed")
        self.logger.info(
            f"Integrity check passed: All columns have correct values and no duplicates for scientific names.\n"
            f"Unique records count: {unique_records_count}")

    @staticmethod
    def assign_species_codes(dataframe, sort_column, start_index=1):
        """
        Assigns unique species codes to the DataFrame, starting from a specified index.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing species data.
            sort_column (str): The name of the column for which to sort by.
            start_index (int): The starting index for numbering.

        Returns:
            pd.DataFrame: DataFrame with the new species_code column.
            int: The next available index for further numbering.
        """
        dataframe = dataframe.copy()  # Work on a copy to avoid modifying the original
        dataframe = dataframe.drop_duplicates()  # Drop duplicates
        dataframe = dataframe.sort_values(by=sort_column)
        dataframe['species_code'] = range(start_index, start_index + len(dataframe))  # Assign codes
        dataframe['species_code'] = dataframe['species_code'].apply(
            lambda x: f"{x:04d}")  # Format codes with leading zeros
        return dataframe, start_index + len(dataframe)  # Return updated dataframe and next start index

    @staticmethod
    def deduplicate_group(group):
        # Access grouping columns via group.name
        park_code, species_code = group.name
        group = group.copy()
        group['park_code'] = park_code
        group['species_code'] = species_code

        # Rule 1: Drop records where 'is_protected' is True if there's a False
        if group['is_protected'].nunique() > 1:
            group = group[group['is_protected'] == False]
        if len(group) == 1:
            return group.iloc[0]

        # Rule 2: Drop 'Unknown' in specified columns if other values exist
        cols = ['occurrence', 'nativeness', 'abundance', 'seasonality']
        for col in cols:
            if 'Unknown' in group[col].values and group[col].nunique() > 1:
                group = group[group[col] != 'Unknown']
                if len(group) == 1:
                    return group.iloc[0]

        # Rule 3: Drop 'In Review' if 'Approved' exists
        if group['record_status'].nunique() > 1:
            if 'Approved' in group['record_status'].values:
                group = group[group['record_status'] == 'Approved']
        if len(group) == 1:
            return group.iloc[0]

        # Rule 4: Arbitrarily drop duplicates if they are identical
        return group.iloc[0]

    def generate_data_dictionary(self, *species_dataframes):
        """
        Generates a full data dictionary from 'all_records' and a list of species DataFrames.

        For 'all_records', it includes column names, data types, and unique values for specific columns.
        For each species DataFrame, it includes column names and data types.

        Args:
            *species_dataframes (pd.DataFrame): Variable length argument list of species DataFrames.

        Returns:
            dict: A comprehensive data dictionary with column names, data types, and unique values.
        """
        data_dictionary = {}

        # Process 'all_records' with hardcoded handling
        all_records = self.all_records
        records_info = {}
        columns_to_log = ['occurrence', 'nativeness', 'abundance', 'seasonality', 'conservation_status']
        for col in all_records.columns:
            col_dtype = str(all_records[col].dtype)
            if col in columns_to_log and isinstance(all_records[col].dtype, pd.CategoricalDtype):
                records_info[col] = {
                    'data_type': 'category',
                    'unique_values': all_records[col].cat.categories.tolist()
                }
            else:
                # Just capture the data type for other columns
                records_info[col] = {
                    'data_type': col_dtype
                }
        data_dictionary['all_records'] = records_info

        # Process each species DataFrame passed in the list
        caller_locals = inspect.currentframe().f_back.f_locals
        for species_df in species_dataframes:
            species_name = [name for name, val in caller_locals.items() if val is species_df][0]
            species_info = {}

            for col in species_df.columns:
                col_dtype = str(species_df[col].dtype)
                if pd.api.types.is_bool_dtype(species_df[col]):
                    species_info[col] = {
                        'data_type': 'bool'
                    }
                elif pd.api.types.is_object_dtype(species_df[col]):
                    species_info[col] = {
                        'data_type': 'object'
                    }
                else:
                    species_info[col] = {
                        'data_type': col_dtype
                    }

            data_dictionary[species_name] = species_info

        # Log the full data dictionary
        self.logger.info("Data Dictionary Generated:")
        for df_name, info in data_dictionary.items():
            self.logger.info(f"{df_name}:")
            for col, details in info.items():
                self.logger.info(f"  Column: {col}, Data Type: {details['data_type']}, "
                                 f"Unique Values: {details.get('unique_values', 'N/A')}")

        return data_dictionary

    def select_random_indices(self, count=10):
        """
        Randomly selects a specified number of indices from the concatenated DataFrame.

        Args:
            count (int): The number of random indices to select.

        Returns:
            dict: A dictionary of selected indices and their associated data.
        """
        # Ensure the random selection does not exceed the available indices
        if count > len(self.records):
            raise ValueError("The count of indices requested exceeds the available indices in the DataFrame.")

        random.seed(42)
        selected_indices = random.sample(list(self.records.index), count)
        selected_records = self.records.loc[selected_indices]
        selected_data_dict = {
            idx: {
                'order': row['order'],
                'family': row['family'],
                'scientific_name': row['scientific_name'],
                'common_names': row['common_names'],
            } for idx, row in selected_records.iterrows()
        }
        self.logger.info("Random Indices for testing:")
        for i, data in selected_data_dict.items():
            self.logger.info(f"{i}: {data}")

        return selected_data_dict

