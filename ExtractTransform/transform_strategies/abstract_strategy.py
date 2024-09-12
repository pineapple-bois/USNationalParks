from abc import ABC, abstractmethod
import pandas as pd
import logging

from ExtractTransform.utils import DataFrameUtils


class TransformStrategy(ABC):
    """
    Abstract base class for transformation strategies for different categories.

    This class defines the interface for transformation strategies that handle
    category-specific data transformations in a DataFrame. Subclasses must implement
    methods for applying transformations and creating category-specific columns.
    """
    @abstractmethod
    def apply_transformations(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Apply specific transformations to the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to transform.
            logger: A logger instance for recording the transformation process.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return dataframe

    @abstractmethod
    def create_category_columns(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """
        Creates category-specific columns in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to modify.
            logger: A logger instance for recording the column creation process.

        Returns:
            pd.DataFrame: The DataFrame with additional category-specific columns.
        """
        pass