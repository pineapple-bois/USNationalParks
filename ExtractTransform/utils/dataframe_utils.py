import os
import logging
import yaml
import pickle
import pandas as pd


class DataFrameUtils:
    @staticmethod
    def save_dataframe_to_csv(df, directory, filename, logger):
        """Saves the given DataFrame to a CSV file in the specified directory with error handling and logging."""
        backup_dir = directory
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup_file_path = os.path.join(backup_dir, filename)
            df.to_csv(backup_file_path, index=True)

            # Check if the file was created and log success
            if os.path.exists(backup_file_path):
                logger.info(f"Data successfully saved to {backup_file_path}\n")
            else:
                raise FileNotFoundError(f"File {backup_file_path} was not found after saving.")

        except Exception as e:
            logger.error(f"Error occurred while saving data: {e}")
            raise


    @staticmethod
    def setup_logger(name, log_filename):
        """Configures a logger with a specified name and log file."""
        log_dir = os.path.join(os.getcwd(), 'Pipeline/Logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to prevent adding duplicates
        if not logger.handlers:
            log_file = os.path.join(log_dir, log_filename)
            handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


    @staticmethod
    def save_dict_to_yaml(data, directory, filename, logger):
        """
        Saves a Python dictionary to a YAML file in a specified directory.

        Args:
            data (dict): The dictionary to save.
            directory (str): The directory where the YAML file will be saved.
            filename (str): The name of the YAML file.
            logger (logging.Logger): Logger for logging messages.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w') as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)

            # Check if the file was created and log success
            if os.path.exists(filepath):
                logger.info(f"Dictionary successfully saved to {filepath}\n"
                            f"Review contents carefully and ensure all data is correct")
            else:
                raise FileNotFoundError(f"File {filepath} was not found after saving.")

        except Exception as e:
            logger.error(f"Error occurred while saving dictionary to YAML: {e}")
            raise


    @staticmethod
    def load_dict_from_yaml(filepath):
        """
        Loads a Python dictionary from a YAML file.

        Args:
            filepath (str): The file path of the YAML file to load.

        """
        try:
            with open(filepath, 'r') as file:
                data = yaml.safe_load(file)
                return data
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error occurred while loading YAML file {filepath}: {e}")
            raise


    @staticmethod
    def verify_dataset_integrity(dataframe: pd.DataFrame, logger) -> pd.DataFrame or bool:
        """
        Verifies the integrity of the dataset by checking for NaN values,
        dropping duplicates, and ensuring the count of unique scientific names
        matches the count of unique records. Logs the results of these checks.

        Returns:
            bool: True if integrity checks pass, otherwise raises an exception or
                  returns a DataFrame of discrepancies for manual update.
        """
        # Check for NaN values in the dataframe
        if dataframe.isna().any().any():
            nan_counts = dataframe.isna().sum()
            nan_fields = nan_counts[nan_counts > 0].index.tolist()
            logger.error(f"Integrity check failed: Found NaN values in fields: {nan_fields}")
            raise ValueError("Dataset contains NaN values. Please address missing data before proceeding.")

        # Drop duplicates and count unique records
        species = dataframe[['order', 'family', 'scientific_name', 'common_names']].drop_duplicates()
        unique_records_count = species.shape[0]
        unique_sci_names_count = species['scientific_name'].nunique()

        # Compare the counts
        if unique_sci_names_count != unique_records_count:
            logger.error(
                f"Integrity check failed: Mismatch between unique scientific names ({unique_sci_names_count}) "
                f"and unique records ({unique_records_count})."
            )

            # Find discrepancies for manual review
            discrepancies = species.groupby('scientific_name').filter(lambda x: len(x) > 1)
            logger.info("Returning discrepancies for manual update.")
            return discrepancies

        # Log success if all checks pass
        logger.info("Dataset integrity verified: No NaN values and counts match.\n")
        return True  # Indicating the integrity check passed


    @staticmethod
    def find_keywords(common_names: str, keywords: list) -> str:
        """
        Finds matching keywords in the common names from a list of keywords.

        Args:
            common_names (str): The common names string to search.
            keywords (list): A list of keywords to find.

        Returns:
            str: A comma-separated string of found keywords.
        """
        if pd.isna(common_names):
            return ''
        matches = {keyword for keyword in keywords if keyword in common_names}
        return ', '.join(matches)


    @staticmethod
    def pickle_data(data, directory, filename, logger):
        """
        Pickles the provided data and exports it to a specified directory.

        Args:
            data: The data to be pickled.
            directory (str): The directory where the pickled file will be saved.
            filename (str): The name of the file to save the pickled data as.
            logger: Logger for logging the saving process.

        Raises:
            Exception: If an error occurs during the pickling process.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
            logger.info(f"Data successfully pickled and saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to pickle data: {e}")
            raise