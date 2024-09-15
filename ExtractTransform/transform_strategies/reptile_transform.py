import pandas as pd
import logging

from ExtractTransform.utils import DataFrameUtils
from ExtractTransform.transform_strategies.abstract_strategy import TransformStrategy


class ReptileTransformStrategy(TransformStrategy):
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        logger.info("APPLYING REPTILE-SPECIFIC TRANSFORMATIONS:\n")
        dataframe = self._correct_discrepancies(dataframe, logger)

        return dataframe

    @staticmethod
    def update_taxonomy(row, taxonomy_info):
        species = row['scientific_name']  # Assuming the species column exists
        if species in taxonomy_info:
            row['order'] = taxonomy_info[species]['order']
            row['family'] = taxonomy_info[species]['family']
        return row

    @staticmethod
    def _correct_discrepancies(dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        # Dictionary with species as keys and corresponding 'order' and 'family' values
        taxonomy_info = {
            'Lampropeltis getulus californiae': {'order': 'Squamata', 'family': 'Colubridae'},
            'Hypsiglena chlorophaea deserticola': {'order': 'Squamata', 'family': 'Colubridae'},
            'Plestiodon gilberti rubricaudatus': {'order': 'Squamata', 'family': 'Scincidae'},
            'Rena humilis cahuilae': {'order': 'Squamata', 'family': 'Leptotyphlopidae'},
            'Rena humilis humilis': {'order': 'Squamata', 'family': 'Leptotyphlopidae'}
        }
        nan_records = dataframe[dataframe['order'].isna() | dataframe['family'].isna()]
        assert nan_records.shape[0] == 7, "Expected 7 records where 'order', 'family' is NaN"

        # Apply the update_taxonomy method with the taxonomy_info dictionary
        dataframe = dataframe.apply(lambda row: ReptileTransformStrategy.update_taxonomy(row, taxonomy_info), axis=1)
        # Drop rows where 'order' or 'family' is still NaN
        dataframe = dataframe.dropna(subset=['order', 'family'])

        # Verify integrity using the utility function.
        integrity_check_result = DataFrameUtils.verify_dataset_integrity(dataframe, logger)
        if isinstance(integrity_check_result, pd.DataFrame):
            logger.warning("Further discrepancies found after corrections, returning for manual review.")
            return integrity_check_result

        return dataframe

    def create_category_columns(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        pass