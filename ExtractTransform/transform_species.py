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
        self.records_dropped = 0
        extract_species = ExtractSpecies()
        self.dataframe = extract_species.dataframe

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

        self.logger.info(f"DataFrame created for category '{self.category}' with shape: {self.dataframe.shape}")

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

    def _apply_category_transformations(self):
        """Applies the appropriate transformation strategy based on the category."""
        self.dataframe = self._remove_genus_only_records()
        self.logger.info(f"Processing Records with no 'common_names':")
        self.dataframe = self._process_no_common_names(condition=2)
        self.logger.info(f"Processing Records with multiple 'common_names':")
        self.dataframe = self._process_multiple_common_names(condition=2)
        self.logger.info(f"Processing Records containing subspecies (Genus species subspecies):")
        self.dataframe = self._process_subspecies()
        self.logger.info(f"Processing Records where 'common_names' is NaN:")
        self.dataframe = self._update_records_nan_common_names()

        # Apply extra methods base on strategy
        strategy = TransformStrategyFactory.get_strategy(self.category)
        self.dataframe = strategy.apply_transformations(self.dataframe, self.logger)
        self.logger.info(f"Total records dropped: {self.records_dropped}")


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
        # Add bird-specific transformations here
        return dataframe



# Concrete Strategy for Mammals
class MammalTransformStrategy(TransformStrategy):
    """
    Transformation strategy for Mammal category.
    """
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        logger.info("Applying mammal-specific transformations.\n")
        # Add mammal-specific transformations here
        return dataframe
