import os
import logging
import pandas as pd
import re
import requests
from pandas.api.types import CategoricalDtype

SPECIES_DATA_URL = "https://api.github.com/repos/pineapple-bois/USNationalParks/contents/DATA/Masters/species.csv"
VALID_CATEGORIES = [
    'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish', 'Vascular Plant',
    'Spider/Scorpion', 'Insect', 'Invertebrate', 'Fungi', 'Nonvascular Plant',
    'Crab/Lobster/Shrimp', 'Slug/Snail', 'Algae'
]


class ExtractSpecies:
    """
    ExtractSpecies class handles the loading, cleaning, transformation, and management of species data,
    allowing for filtering by a specified category.

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

    Methods:
        get_dataframe() -> pd.DataFrame:
            Returns the subset DataFrame for the specified category.

        get_logger() -> logging.Logger:
            Returns the logger object.

        _load_and_clean_data() -> pd.DataFrame:
            Loads species data from the specified URL, performs initial cleaning,
            and returns the cleaned DataFrame.

        _remove_false_data() -> pd.DataFrame:
            Filters out records where 'record_status' is not 'Approved' or 'In Review',
            saves these records to a CSV file, shifts relevant column data,
            and logs the actions.

        _transform_dataframe() -> pd.DataFrame:
            Transforms the entire DataFrame by setting categorical and boolean columns,
            and simplifying the 'seasonality' column.

        _subset_category() -> pd.DataFrame:
            Filters the transformed DataFrame to include only the specified category.
    """
    def __init__(self, category):
        if not isinstance(category, str):
            raise TypeError("Category must be a string.")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Valid options are: {', '.join(VALID_CATEGORIES)}")

        self.category = category
        self.logger = self._setup_logger()
        try:
            self.dataframe = self._load_and_clean_data()  # Initial loading and cleaning
        except Exception as e:
            self.logger.error(f"Error loading and cleaning data: {e}")
            raise

        self.logger.info(f"Initial DataFrame shape: {self.dataframe.shape}")

        try:
            self.dataframe = self._clean_non_standard_rows() # removing non-standard chars
        except Exception as e:
            self.logger.error(f"Error removing non-standard characters: {e}")
            raise

        try:
            self.dataframe = self._remove_false_data()
        except Exception as e:
            self.logger.error(f"Error removing false data: {e}")
            raise

        try:
            self.dataframe = self._transform_dataframe()  # Transformation
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise

        self.logger.info(f"Data types after transformation:\n{self.dataframe.dtypes}")

        try:
            self.dataframe = self._subset_category()  # Subset by category
        except Exception as e:
            self.logger.error(f"Error subsetting data: {e}")
            raise

        self.logger.info(f"Subset DataFrame for category '{self.category}' shape: {self.dataframe.shape}")

    def save_dataframe_to_csv(self, df, directory, filename):
        """Saves the given DataFrame to a CSV file in the specified directory with error handling and logging."""
        backup_dir = directory
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup_file_path = os.path.join(backup_dir, filename)
            df.to_csv(backup_file_path, index=True)

            # Check if the file was created and log success
            if os.path.exists(backup_file_path):
                self.logger.info(f"Data successfully saved to {backup_file_path}")
            else:
                raise FileNotFoundError(f"File {backup_file_path} was not found after saving.")

        except Exception as e:
            self.logger.error(f"Error occurred while saving data: {e}")
            raise

    def _setup_logger(self):
        """Configures a logger specific to the category."""
        log_dir = os.path.join(os.getcwd(), 'Logs')
        os.makedirs(log_dir, exist_ok=True)
        formatted_category = self.category.lower().replace(" ", "_").replace("/", "_")
        logger = logging.getLogger(self.category)
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to prevent adding duplicates
        if not logger.handlers:
            log_file = os.path.join(log_dir, f'transformation_{formatted_category}.log')
            handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_and_clean_data(self):
        """Loads species data from the specified URL and performs initial cleaning."""
        try:
            response = requests.get(SPECIES_DATA_URL)
            response.raise_for_status()
            file_info = response.json()
            download_url = file_info['download_url']
            df = pd.read_csv(download_url, low_memory=False)
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            self.logger.error(f"An error occurred: {err}")
            raise

        # Perform initial cleaning
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        df.rename(columns={'unnamed:_13': 'unnamed'}, inplace=True)
        df['park_name'] = df['park_name'].str.replace(r' National Park.*', '', case=False, regex=True)
        df['park_code'] = df['species_id'].str.split('-').str[0]
        cols = list(df.columns)
        cols.insert(1, cols.pop(cols.index('park_code')))
        df = df[cols]
        return df

    @staticmethod
    def has_non_standard_chars(value):
        if pd.isna(value):  # Check for NaN values
            return False
        # Check for non-standard characters using regex
        return bool(re.search(r"[^a-zA-Z0-9\s,.()\-_'’ʻ`´]", str(value)))

    @staticmethod
    def clean_non_standard_chars(value):
        if pd.isna(value):  # Check for NaN values
            return value
        value = re.sub(r"/", ", ", str(value))
        value = re.sub(r";", ",", str(value))
        value = re.sub(r"[^a-zA-Z0-9\s,.()\-&'’ʻ]", "", value)
        return value

    def _clean_non_standard_rows(self):
        df = self.dataframe.copy()
        columns = ['scientific_name', 'common_names']

        # Apply the function column-wise to check for non-standard characters in each column
        columns_with_issues = []
        for column in columns:
            if df[column].apply(self.has_non_standard_chars).any():
                columns_with_issues.append(column)

        # Filter rows where any of the columns have non-standard characters
        df_with_non_standard_chars = df[df[columns_with_issues].apply(
            lambda row: row.apply(self.has_non_standard_chars).any(), axis=1)].copy()
        df_to_save = df_with_non_standard_chars[['category', 'scientific_name', 'common_names']]
        self.logger.info(f"Found {len(df_to_save)} records with non-standard characters.")
        self.save_dataframe_to_csv(df_to_save, "BackupData", "nonstandard_chars.csv")

        # Apply the cleaning function in place
        for column in columns_with_issues:
            df_with_non_standard_chars[column] = df_with_non_standard_chars[column].apply(
                self.clean_non_standard_chars)
        self.dataframe = df
        return df

    def _remove_false_data(self):
        """
        Filters out records where 'record_status' is not 'Approved' or 'In Review',
        saves these records to a CSV file, shifts relevant column data, and logs the actions.
        """
        df = self.dataframe.copy()
        # Filter records where 'record_status' is not 'Approved' or 'In Review'
        non_standard_status_records = df[~df['record_status'].isin(['Approved', 'In Review'])].copy()
        self.logger.info(f"Found {len(non_standard_status_records)} records with non-standard 'record_status'.")
        self.save_dataframe_to_csv(non_standard_status_records, "BackupData", "incorrect_records.csv")

        # Indices that need correction based on 'non_standard_status_records'
        indices_to_shift = non_standard_status_records.index
        start_pos = non_standard_status_records.columns.get_loc('record_status')

        # Shift relevant columns to the left and fill with NaN
        # Use .loc with .copy() to ensure a deep copy is modified
        shifted_values = non_standard_status_records.iloc[:, start_pos + 1:].copy().values
        non_standard_status_records.loc[
            indices_to_shift, non_standard_status_records.columns[start_pos:-1]] = shifted_values
        non_standard_status_records.loc[indices_to_shift, non_standard_status_records.columns[-1]] = pd.NA

        # Apply changes back to the original dataframe
        df.loc[indices_to_shift] = non_standard_status_records.loc[indices_to_shift]

        # Assert to ensure 'unnamed' has only NaN values and drop the column
        assert df['unnamed'].isna().all(), "The 'unnamed' column contains non-NaN values."
        df = df.drop(columns=['unnamed'])
        self.dataframe = df
        return df

    @staticmethod
    def simplify_seasonality(value):
        """
        Simplifies the seasonality values to a single keyword based on priority.
        """
        priority_keywords = ['Winter', 'Summer', 'Breeder', 'Migratory', 'Resident', 'Vagrant']
        for keyword in priority_keywords:
            if keyword in value:
                return keyword
        return 'Unknown'

    def _transform_dataframe(self):
        """
        Transforms the entire DataFrame by setting categorical and boolean columns,
        and simplifying the 'seasonality' column.
        """
        df = self.dataframe.copy()
        df['seasonality'] = df['seasonality'].fillna('Unknown')
        df['seasonality'] = df['seasonality'].apply(self.simplify_seasonality)

        # Define fill values and category orders
        fill_values = {
            'conservation_status': 'Least Concern',
            'abundance': 'Unknown',
            'nativeness': 'Unknown',
            'occurrence': 'Not Confirmed'
        }
        conservation_status_order = ['Least Concern', 'Species of Concern', 'In Recovery', 'Under Review',
                                     'Threatened', 'Proposed Endangered', 'Endangered']
        abundance_order = ['Rare', 'Uncommon', 'Unknown', 'Occasional', 'Common', 'Abundant']
        nativeness_order = ['Not Native', 'Unknown', 'Native']
        record_status_order = ['In Review', 'Approved']
        occurrence_order = ['Not Present (False Report)', 'Not Present (Historical Report)',
                            'Not Present', 'Not Confirmed', 'Present']

        # Fill NaN values and set categorical data types
        for column, fill_value in fill_values.items():
            df[column] = df[column].fillna(fill_value)

        # Set categorical types with defined order
        df['record_status'] = pd.Categorical(df['record_status'], categories=record_status_order, ordered=True)
        df['occurrence'] = pd.Categorical(df['occurrence'], categories=occurrence_order, ordered=True)
        df['nativeness'] = pd.Categorical(df['nativeness'], categories=nativeness_order, ordered=True)
        df['abundance'] = pd.Categorical(df['abundance'], categories=abundance_order, ordered=True)
        df['conservation_status'] = pd.Categorical(df['conservation_status'], categories=conservation_status_order,
                                                   ordered=True)

        # Add a boolean column 'is_protected'
        df['is_protected'] = df['conservation_status'] != 'Least Concern'

        self.dataframe = df
        return df

    def _subset_category(self):
        """Filters the transformed DataFrame to include only the specified category."""
        # Filter the DataFrame by the specified category
        df = self.dataframe[self.dataframe['category'] == self.category]
        return df

    def get_dataframe(self):
        """Returns the subset DataFrame for the specified category."""
        return self.dataframe

    def get_logger(self):
        """Returns the logger object."""
        return self.logger

