import pandas as pd
from abc import ABC, abstractmethod
from utils import DataFrameUtils, DataFrameTransformation
from extract_species import ExtractSpecies

# Add to below as required as and when abstract base methods written
VALID_CATEGORIES = [
    'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish'
]


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

    Methods:
        __init__(category: str):
            Initializes the TransformSpecies class, sets up logging for the specified category,
            loads and subsets the species data for further transformations.

        _subset_category() -> pd.DataFrame:
            Filters the transformed DataFrame to include only the specified category.

        _remove_genus_only_records():
            Identifies and removes records that contain only the genus name in the 'scientific_name' field,
            logs the actions, and saves these records to a CSV file.

        _apply_category_transformations():
            Applies the appropriate transformation strategy based on the category.
    """
    def __init__(self, category):
        if not isinstance(category, str):
            raise TypeError("Category must be a string.")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Valid options are: {', '.join(VALID_CATEGORIES)}")

        self.category = category
        formatted_category = self.category.lower().replace(" ", "_").replace("/", "_")
        self.logger = DataFrameUtils.setup_logger(f'transform_{formatted_category}',
                                                  f'transformation_{formatted_category}.log')

        extract_species = ExtractSpecies()
        self.dataframe = extract_species.dataframe

        try:
            self.dataframe = self._subset_category()  # Subset by category
        except Exception as e:
            self.logger.error(f"Error subsetting data for category '{self.category}': {e}")
            raise

        try:
            self.dataframe = self._remove_genus_only_records()
        except Exception as e:
            self.logger.error(f"Error removing Genus only records for category '{self.category}': {e}")
            raise

        try:
            self._apply_category_transformations()
        except Exception as e:
            self.logger.error(f"Error applying transformations for category '{self.category}': {e}")
            raise

        self.logger.info(f"DataFrame created for category '{self.category}' with shape: {self.dataframe.shape}")

    def _subset_category(self):
        """Filters the transformed DataFrame to include only the specified category."""
        # Filter the DataFrame by the specified category
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
        self.logger.info(f"{len(df_to_save)} records dropped.\n")

        df_copy.drop(df_to_save.index, inplace=True)
        self.dataframe = df_copy
        return self.dataframe

    def _apply_category_transformations(self):
        """Applies the appropriate transformation strategy based on the category."""
        strategy = TransformStrategyFactory.get_strategy(self.category)
        self.dataframe = strategy.apply_transformations(self.dataframe, self.logger)


# Abstract Base Class
class TransformStrategy(ABC):
    """
    Abstract base class for transformation strategies for different categories.
    """
    @abstractmethod
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """Apply specific transformations to the DataFrame."""
        return dataframe


# Factory Class
class TransformStrategyFactory:
    """
    Factory to create transformation strategies based on the category.
    """
    _strategies = {
        'Bird': lambda: BirdTransformStrategy(),
        'Mammal': lambda: MammalTransformStrategy(),
        # Other mappings here when scaling
    }

    @staticmethod
    def get_strategy(category: str) -> TransformStrategy:
        try:
            return TransformStrategyFactory._strategies[category]()
        except KeyError:
            raise ValueError(f"No transformation strategy defined for category: {category}")


# Concrete Strategy for Birds
class BirdTransformStrategy(TransformStrategy):
    """
    Transformation strategy for Bird category.
    """
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        logger.info("Applying bird-specific transformations:\n")
        dataframe = self._process_no_common_names(dataframe, logger, condition=2)
        dataframe = self._process_multiple_common_names(dataframe, logger, condition=2)
        dataframe = self._process_subspecies(dataframe, logger)
        dataframe = self._update_records_nan_common_names(dataframe, logger)
        return dataframe

    def _process_no_common_names(self, dataframe: pd.DataFrame, logger, condition):
        """
        Processes scientific names with no associated common names based on the given condition.
        Saves identified records and updates common names.
        """
        # Get records with no common names based on the condition
        results = DataFrameTransformation.process_scientific_names(dataframe, condition=condition)
        no_common_names = results['no_common_names']
        no_common_names_records = dataframe[dataframe['scientific_name'].isin(no_common_names)]
        logger.info(f"Found {len(no_common_names)} scientific names with no associated common names for condition {condition}.")
        DataFrameUtils.save_dataframe_to_csv(no_common_names_records, f"BackupData/Bird",
                                             f"no_common_names_condition_{condition}.csv", logger)

        # Fuzzy match and update
        dataframe = DataFrameTransformation.fuzzy_match_and_update(
            dataframe, no_common_names, 'common_names', logger
        )
        return dataframe

    def _process_multiple_common_names(self, dataframe: pd.DataFrame, logger, condition):
        """
        Processes scientific names with multiple associated common names based on the given condition.
        Saves identified records and standardizes common names.
        """
        # Get records with multiple common names based on the condition
        results = DataFrameTransformation.process_scientific_names(dataframe, condition=condition)
        multiple_common_names = results['multiple_common_names']
        logger.info(f"Found {len(multiple_common_names)} scientific names with multiple common names for condition {condition}.")

        multi_sci_names_list = [sci_name for sci_name, counts in multiple_common_names]
        multi_common_names_records = dataframe[dataframe['scientific_name'].isin(multi_sci_names_list)]
        DataFrameUtils.save_dataframe_to_csv(multi_common_names_records, f"BackupData/Bird",
                                             f"multiple_common_names_condition_{condition}.csv", logger)

        # Standardize the common names for identified records
        if condition == 3:
            # Use the subspecies standardization method if processing subspecies
            dataframe = DataFrameTransformation.standardize_names(
                dataframe, multiple_common_names,
                DataFrameTransformation.standardize_common_names_subspecies, logger
            )
        else:
            dataframe = DataFrameTransformation.standardize_names(
                dataframe, multiple_common_names,
                DataFrameTransformation.standardize_common_names, logger
            )

        return dataframe

    def _process_subspecies(self, dataframe: pd.DataFrame, logger):
        """
        Handles subspecies records by processing scientific names with no common names and
        those with multiple common names specifically for subspecies.
        """
        # Process subspecies with no common names
        dataframe = self._process_no_common_names(dataframe, logger, condition=3)
        # Process subspecies with multiple common names
        dataframe = self._process_multiple_common_names(dataframe, logger, condition=3)
        return dataframe

    def _update_records_nan_common_names(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Identifies records with NaN in 'common_names' and updates them using fuzzy matching.
        """
        nan_common_names_records = dataframe[dataframe['common_names'].isna()]
        nan_sci_names = nan_common_names_records['scientific_name'].unique().tolist()
        logger.info(f"Found {nan_common_names_records.shape[0]} scientific names with no associated common names.")
        DataFrameUtils.save_dataframe_to_csv(nan_common_names_records, f"BackupData/Bird",
                                             "nan_common_names.csv", logger)

        dataframe = DataFrameTransformation.fuzzy_match_and_update(
            dataframe, nan_sci_names, 'common_names', logger
        )
        return dataframe


# Concrete Strategy for Mammals
class MammalTransformStrategy(TransformStrategy):
    """
    Transformation strategy for Mammal category.
    """
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        logger.info("Applying mammal-specific transformations.")
        # Add mammal-specific transformations here
        return dataframe
