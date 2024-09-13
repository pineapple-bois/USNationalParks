import pandas as pd
import re
import requests
from pandas.api.types import CategoricalDtype
from ExtractTransform.utils import DataFrameUtils


SPECIES_DATA_URL = "https://api.github.com/repos/pineapple-bois/USNationalParks/contents/DATA/Masters/species.csv"


class ExtractSpecies:
    """
    ExtractSpecies class handles the loading, cleaning, transformation, and management of species data.

    Attributes:
        dataframe (pd.DataFrame): The transformed DataFrame containing all species data.
        logger (logging.Logger): A logger object configured to log information about data processing.

    Public Methods:
        assert_dataframe_integrity():
            Verifies the integrity of the DataFrame by ensuring that essential columns have the correct data types
            and are free from NaN values. This method is publicly accessible and is intended to confirm that the
            DataFrame meets specific integrity requirements for critical columns.


    Private Methods:
        _load_and_clean_data() -> pd.DataFrame:
            Loads species data from the specified URL, performs initial cleaning (e.g., renaming columns,
            extracting park codes), and returns the cleaned DataFrame.

        _remove_false_data() -> pd.DataFrame:
            Filters out records where 'record_status' is not 'Approved' or 'In Review', saves these records
            to a CSV file, shifts relevant column data, and logs the actions.

        _clean_non_standard_rows() -> pd.DataFrame:
            Identifies and cleans rows with non-standard characters in specified columns, logs the
            problematic records, and saves them to a CSV file.

        _transform_dataframe() -> pd.DataFrame:
            Transforms the entire DataFrame by setting categorical and boolean columns, simplifying the
            'seasonality' column, and adding an 'is_protected' column.

    Static Methods:
        has_non_standard_chars(value: Any) -> bool:
            Checks if a given value contains non-standard characters using regular expressions.

        clean_non_standard_chars(value: Any) -> str:
            Cleans non-standard characters from a given value by replacing and removing unwanted symbols.

        simplify_seasonality(value: str) -> str:
            Simplifies the seasonality values to a single keyword based on priority (e.g., 'Winter', 'Summer').
    """
    def __init__(self):
        self.logger = DataFrameUtils.setup_logger('extract_species', 'extract_species.log')
        try:
            self.dataframe = self._load_and_clean_data()  # Initial loading and cleaning
        except Exception as e:
            self.logger.error(f"Error loading and cleaning data: {e}")
            raise

        self.logger.info(f"Initial DataFrame shape: {self.dataframe.shape}")

        try:
            self.dataframe = self._remove_false_data()
        except Exception as e:
            self.logger.error(f"Error removing false data: {e}")
            raise

        try:
            self.dataframe = self._clean_non_standard_rows() # removing non-standard chars
        except Exception as e:
            self.logger.error(f"Error removing non-standard characters: {e}")
            raise

        try:
            self.dataframe = self._transform_dataframe()  # Transformation
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise

        print(f"Species master data created.\nShape: {self.dataframe.shape}")
        self.logger.info(f"Final DataFrame shape: {self.dataframe.shape}\n"
                         f"Data types after transformation:\n{self.dataframe.dtypes}")
        DataFrameUtils.pickle_data(self.dataframe,
                                   "Pipeline/FinalData",
                                   "species_master.pkl",
                                   self.logger
                                   )

    def _load_and_clean_data(self):
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

    def _remove_false_data(self):
        df = self.dataframe.copy()
        # Filter records where 'record_status' is not 'Approved' or 'In Review'
        non_standard_status_records = df[~df['record_status'].isin(['Approved', 'In Review'])].copy()
        self.logger.info(f"Found {len(non_standard_status_records)} records with non-standard 'record_status'.")
        DataFrameUtils.save_dataframe_to_csv(non_standard_status_records,
                                             "Pipeline/BackupData",
                                             "incorrect_records.csv",
                                             self.logger)

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

        # Identify columns that have non-standard characters
        columns_with_issues = [column for column in columns if df[column].apply(self.has_non_standard_chars).any()]

        # Filter rows where any of the columns have non-standard characters
        df_with_non_standard_chars = df[df[columns_with_issues].apply(
            lambda row: row.apply(self.has_non_standard_chars).any(), axis=1)].copy()

        df_to_save = df_with_non_standard_chars[['category', 'scientific_name', 'common_names']]
        self.logger.info(f"Found {len(df_to_save)} records with non-standard characters.")
        DataFrameUtils.save_dataframe_to_csv(df_to_save,
                                             "Pipeline/BackupData",
                                             "nonstandard_chars.csv",
                                             self.logger
                                             )

        # Clean the non-standard characters in place in the copy
        for column in columns_with_issues:
            df_with_non_standard_chars[column] = df_with_non_standard_chars[column].apply(
                self.clean_non_standard_chars)

        df.update(df_with_non_standard_chars)
        self.logger.info(f"\n\nDropping ill-formatted, ambiguous record @ index 30773. "
                         f"\n\n'common_names' field:\n{df.loc[30773, 'common_names']}\n\n")
        df.drop(index=30773, inplace=True) # Dropping a particularly ill-formatted record
        self.dataframe = df
        return df

    @staticmethod
    def simplify_seasonality(value):
        priority_keywords = ['Winter', 'Summer', 'Breeder', 'Migratory', 'Resident', 'Vagrant']
        for keyword in priority_keywords:
            if keyword in value:
                return keyword
        return 'Unknown'

    def _transform_dataframe(self):
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

        # Verify if all 'conservation_status' entries are within the defined categories
        df['conservation_status'] = df['conservation_status'].apply(
            lambda x: x if x in conservation_status_order else 'Least Concern'
        )
        df['conservation_status'] = pd.Categorical(df['conservation_status'], categories=conservation_status_order,
                                                   ordered=True)

        # Add a boolean column 'is_protected'
        df['is_protected'] = df['conservation_status'] != 'Least Concern'

        self.dataframe = df
        return df

    def assert_dataframe_integrity(self):
        """
        Asserts that the DataFrame meets specific integrity requirements:
        - Certain columns have defined data types.
        - No NaN values in key columns.

        Expected Data Types:
            - record_status: category
            - occurrence: category
            - nativeness: category
            - abundance: category
            - seasonality: object
            - conservation_status: category
            - is_protected: bool

        Raises:
            AssertionError: If any of the conditions are not met.
        """
        expected_types = {
            'record_status': 'category',
            'occurrence': 'category',
            'nativeness': 'category',
            'abundance': 'category',
            'seasonality': 'object',
            'conservation_status': 'category',
            'is_protected': 'bool'
        }

        # Check column types
        for column, expected_type in expected_types.items():
            actual_type = str(self.dataframe[column].dtype)
            assert actual_type == expected_type, \
                f"Column '{column}' expected type '{expected_type}', but got '{actual_type}'."

        # Check for NaN values
        nan_counts = self.dataframe[expected_types.keys()].isna().sum()
        nan_fields = nan_counts[nan_counts > 0].index.tolist()
        assert not nan_fields, f"Columns with NaN values: {nan_fields}"

        print("DataFrame integrity check passed:\nAll columns have correct types and no NaN values in critical fields.")
