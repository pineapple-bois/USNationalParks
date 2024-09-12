import pandas as pd
import logging

from ExtractTransform.utils import DataFrameUtils
from ExtractTransform.transform_strategies.abstract_strategy import TransformStrategy


class BirdTransformStrategy(TransformStrategy):
    """
    Transformation strategy for Bird category.

    This class implements bird-specific transformations, particularly focused on
    identifying birds of prey by using common names and scientific families associated
    with birds of prey. It also handles corrections for known data discrepancies.

    Birds of Prey Scientific Families:
        - Accipitridae (Hawks, Eagles, and relatives)
        - Falconidae (Falcons)
        - Harpagiidae (Harriers)
        - Pandionidae (Ospreys)
        - Accipitridae (Kites)
        - Cathartidae (New World Vultures)
        - Buteo (Buzzards and Buteos)
        - Accipiter (Goshawks and Accipiters)
        - Tytonidae (Barn Owls)
        - Strigidae (Typical Owls)
    """

    BIRDS_OF_PREY = [
        "Eagle", "Hawk", "Falcon", "Buzzard", "Harrier", "Kite",
        "Owl", "Osprey", "Vulture", "Condor", "Kestrel", "Buteo",
        "Accipiter", "Caracara"
    ]

    BIRDS_OF_PREY_FAMILIES = [
        "Accipitridae", "Falconidae", "Harpagiidae", "Pandionidae",
        "Cathartidae", "Buteo", "Accipiter", "Tytonidae", "Strigidae"
    ]

    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Applies bird-specific transformations to the DataFrame.

        This includes correcting known discrepancies and creating columns to
        identify birds of prey based on common names and scientific classifications.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing bird data.
            logger: A logger instance for recording the transformation process.

        Returns:
            pd.DataFrame: The DataFrame with bird-specific amendments applied.
        """
        logger.info("APPLYING BIRD-SPECIFIC TRANSFORMATIONS:\n")
        dataframe = self._correct_discrepancies(dataframe, logger)
        logger.info("Identifying birds of prey:")
        dataframe = self.create_category_columns(dataframe, logger)

        # Future bird-specific transformations can be added here.

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
        # Correct families for specific scientific names based on verified data.
        correct_families = {
            'Phylloscopus borealis': 'Phylloscopidae',
            'Polioptila caerulea': 'Polioptilidae'
        }

        # Update the 'family' column in the DataFrame for the specified scientific names.
        for sci_name, correct_family in correct_families.items():
            updated_count = dataframe.loc[dataframe['scientific_name'] == sci_name, 'family'].size
            dataframe.loc[dataframe['scientific_name'] == sci_name, 'family'] = correct_family
            logger.info(f"Updated family to '{correct_family}' for '{sci_name}' ({updated_count} records).")

        # Verify integrity using the utility function.
        integrity_check_result = DataFrameUtils.verify_dataset_integrity(dataframe, logger)
        if isinstance(integrity_check_result, pd.DataFrame):
            logger.warning("Further discrepancies found after corrections, returning for manual review.")
            return integrity_check_result

        return dataframe

    def create_category_columns(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Creates columns in the DataFrame to identify birds of prey.

        This method identifies birds of prey by checking common names and scientific families
        against predefined lists of known birds of prey. It handles ambiguous cases and updates
        the DataFrame with additional columns for easier identification of these birds.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing bird data.
            logger: A logger instance for recording the process of column creation.

        Returns:
            pd.DataFrame: The DataFrame with new columns added for bird of prey identification.
        """
        dataframe['raptor_group'] = dataframe['common_names'].apply(
            lambda x: DataFrameUtils.find_keywords(x, self.BIRDS_OF_PREY)
        )

        dataframe['raptor_sci_fam'] = dataframe['family'].str.findall(f"({'|'.join(self.BIRDS_OF_PREY_FAMILIES)})")
        dataframe['raptor_sci_fam'] = dataframe['raptor_sci_fam'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else ''
        )

        # Create a subset and flag ambiguous entries where 'raptor_group' is empty
        raptors_df = dataframe[(dataframe['raptor_group'] != '') | (dataframe['raptor_sci_fam'] != '')].copy()
        raptors_df['ambiguous'] = raptors_df['raptor_group'] == ''

        # Handle specific cases for Northern Goshawk and Merlin
        merlin_gyrfalcon_mask = raptors_df['common_names'].str.contains(r'Merlin|Gyrfalcon', case=False, regex=True)
        northern_goshawk_mask = raptors_df['common_names'].str.contains(r'Northern Goshawk', case=False, regex=True)
        raptors_df.loc[merlin_gyrfalcon_mask, 'raptor_common'] = 'Falcon'
        raptors_df.loc[northern_goshawk_mask, 'raptor_common'] = 'Hawk'

        # Log how many records were updated for Merlin|Gyrfalcon
        merlin_gyrfalcon_updated_count = merlin_gyrfalcon_mask.sum()
        northern_goshawk_updated_count = northern_goshawk_mask.sum()
        logger.info(f"Updated 'raptor_common' for {merlin_gyrfalcon_updated_count} Merlin/Gyrfalcon records as 'Falcon'.")
        logger.info(f"Updated 'raptor_common' for {northern_goshawk_updated_count} Northern Goshawk records as 'Hawk'.")

        # Update 'raptor_common' for ambiguous entries based on corrected common names
        ambiguous_result = raptors_df[raptors_df['ambiguous'] == True]
        updated_indices = ambiguous_result.index
        dataframe.loc[updated_indices, 'raptor_group'] = ambiguous_result['raptor_common']
        assert not dataframe.loc[updated_indices, 'raptor_group'].isna().any(), \
            "NaN values present in 'raptor_common' for the specified indices"

        # Handle cases where 'raptor_group' contains multiple entries (specifically: 'Hawk, Owl')
        hawk_owl_indices = dataframe[dataframe['raptor_group'] == 'Hawk, Owl'].index
        dataframe.loc[hawk_owl_indices, 'raptor_group'] = "Owl"

        # Replace empty strings in 'raptor_group' with "N/A" and create boolean column 'is_raptor'
        dataframe['raptor_group'] = dataframe['raptor_group'].replace('', 'Not Applicable')
        dataframe['is_raptor'] = dataframe['raptor_group'].apply(lambda x: x != 'Not Applicable')
        dataframe = dataframe.drop(columns=['raptor_sci_fam'], errors='ignore')

        assert dataframe['is_raptor'].dtype == bool, "'is_raptor' column is not of boolean type."
        assert not dataframe['raptor_group'].isna().any(), "'raptor_group' column contains NaN values."

        columns_order = dataframe.columns.tolist()
        common_names_index = columns_order.index('common_names')
        columns_order.insert(common_names_index + 1, columns_order.pop(columns_order.index('raptor_group')))
        dataframe = dataframe[columns_order]

        logger.info("Bird of prey data created.\n")
        return dataframe
