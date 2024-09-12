import pandas as pd
import logging

from ExtractTransform.utils import DataFrameUtils
from ExtractTransform.transform_strategies.abstract_strategy import TransformStrategy


class MammalTransformStrategy(TransformStrategy):
    """
    Transformation strategy for Mammal category.

    This class implements mammal-specific transformations, particularly focused on
    identifying large (land based) predators using common names and scientific families

    Large predator Scientific Families:
        - Felidae (Cats, big cats)
        - Canidae (Wolves, coyotes, foxes)
        - Ursidae (Bears)
        - Hominidae (Humans)
    """

    MAMMAL_PREDATORS = [
        "Cat", "Big Cat", "Bear", "Wolf", "Coyote", "Human", "Lion", "Cougar",
        "Lynx", "Bobcat", "Jaguar", "Puma", "Panther", "Fox", "Ocelot", "Wildcat"
    ]

    MAMMAL_PREDATOR_FAMILIES = [
        "Felidae", "Canidae", "Ursidae", "Hominidae"
    ]

    # Exclude domesticated/not predator keywords as fallback
    EXCEPTIONS = [
        "Domestic", "Feral", "Squirrel", "Sea Lion", "Dog", "Cat",
        "Cattle", "Goat", "Sheep", "Pig", "Aurochs", "Seal"
    ]


    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        logger.info("APPLYING MAMMAL SPECIFIC TRANSFORMATIONS.\n")
        dataframe = self._correct_discrepancies(dataframe, logger)
        logger.info("Identifying apex predators")
        dataframe = self.create_category_columns(dataframe, logger)


        # Future mammal-specific transformations can be added here.

        return dataframe

    @staticmethod
    def _correct_discrepancies(dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Corrects known discrepancies in the bird dataset.

        Updates specific fields based on manually verified data to ensure the integrity
        of the dataset, particularly focusing on correcting the 'family' field for certain
        scientific names.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing bird data.
            logger: A logger instance for recording the corrections applied.

        Returns:
            pd.DataFrame: The DataFrame with corrected discrepancies. If further
                          discrepancies are detected, it returns a DataFrame for manual review.
        """
        # Verify integrity using the utility function.
        integrity_check_result = DataFrameUtils.verify_dataset_integrity(dataframe, logger)
        if isinstance(integrity_check_result, pd.DataFrame):
            return integrity_check_result

        return dataframe

    def create_category_columns(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Creates columns in the DataFrame to identify apex predators.

        This method identifies apex predators by checking common names and scientific families
        against predefined lists of known predators. It updates the DataFrame with additional
        columns for easier identification of these mammals.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing mammal data.
            logger: A logger instance for recording the process of column creation.

        Returns:
            pd.DataFrame: The DataFrame with new columns added for apex predator identification.
        """

        # Define Big Cats for specific grouping
        BIG_CATS = ["Lynx", "Bobcat", "Cougar", "Jaguar", "Mountain Lion", "Panther", "Ocelot", "Wildcat"]

        # Identify potential predator groups from common names, excluding exceptions
        dataframe['predator_group'] = dataframe['common_names'].apply(
            lambda x: (
                "Big Cat" if any(cat in x for cat in BIG_CATS)
                else DataFrameUtils.find_keywords(x, self.MAMMAL_PREDATORS)
                if not any(exc in x for exc in self.EXCEPTIONS) else ''
            )
        )

        # Identify predator families from the scientific family field
        dataframe['predator_sci_fam'] = dataframe['family'].str.findall(f"({'|'.join(self.MAMMAL_PREDATOR_FAMILIES)})")
        dataframe['predator_sci_fam'] = dataframe['predator_sci_fam'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else ''
        )

        # Flag rows as 'ambiguous_result' where 'predator_group' is empty but 'predator_sci_fam' is not
        dataframe['ambiguous_result'] = (dataframe['predator_group'] == '') & (dataframe['predator_sci_fam'] != '')
        # Log the number of ambiguous cases
        ambiguous_df = dataframe[dataframe['ambiguous_result'] == True]
        logger.info(f"Found {len(ambiguous_df)} ambiguous records where predator status couldn't be clearly identified.")
        unique_ambiguous_names = ambiguous_df['common_names'].unique()
        logger.info(f"{unique_ambiguous_names} removed from list of large predators:")

        dataframe['predator_group'] = dataframe['predator_group'].replace('', 'Not Applicable')
        dataframe['is_large_predator'] = dataframe['predator_group'] != 'Not Applicable'
        assert dataframe['is_large_predator'].dtype == bool, "'is_large_predator' column is not of boolean type."

        unique_large_predators_count = dataframe[dataframe['is_large_predator']]['common_names'].nunique()
        logger.info(f"{unique_large_predators_count} unique large predators found.")

        dataframe = dataframe.drop(columns=['predator_sci_fam', 'ambiguous_result'], errors='ignore')
        columns_order = dataframe.columns.tolist()
        common_names_index = columns_order.index('common_names')
        columns_order.insert(common_names_index + 1, columns_order.pop(columns_order.index('predator_group')))
        dataframe = dataframe[columns_order]

        return dataframe
