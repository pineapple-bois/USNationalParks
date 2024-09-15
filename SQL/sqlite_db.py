import pandas as pd
import sqlite3
import os
import re
import pickle
import requests
from io import StringIO
import logging


# Define file paths
data_dir = '../Pipeline/FinalData'
birds_file = os.path.join(data_dir, 'bird_master.pkl')
mammals_file = os.path.join(data_dir, 'mammal_master.pkl')
records_file = os.path.join(data_dir, 'record_master.pkl')
required_files = [birds_file, mammals_file, records_file]

sqlite_db_path = '../national_parks_lite.db'
csv_url = "https://raw.githubusercontent.com/pineapple-bois/USNationalParks/main/DATA/Masters/parks.csv"


def setup_logger(name, log_filename):
    """Configures a logger with a specified name and log file."""
    log_dir = os.path.join(os.getcwd())
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

logger = setup_logger('sqlite_logger', 'national_parks_lite.log')

# Function to check if required files exist
def check_files_exist(required_files, logger):
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        print("Data files are missing. Please run the data extraction step before loading data into the database.")
        sys.exit(1)  # Exit the script with an error code

# Function to load data from pickle files and insert into SQLite
def load_data_to_sqlite(conn, data, table_name, logger):
    """Load data from DataFrame into SQLite table."""
    try:
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        logger.info(f"{table_name} data loaded into SQLite.")
    except Exception as e:
        logger.error(f"Failed to load {table_name} data into SQLite: {e}")

# Create SQLite database and schema
def create_sqlite_schema(conn):
    """Create tables in the SQLite database."""
    cursor = conn.cursor()

    # Create parks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parks (
            park_code TEXT PRIMARY KEY,
            park_name TEXT,
            state TEXT,
            square_km REAL,
            latitude REAL,
            longitude REAL
        );
    ''')

    # Create birds table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS birds (
            species_code TEXT PRIMARY KEY,
            "order" TEXT NOT NULL,
            family TEXT NOT NULL,
            scientific_name TEXT NOT NULL,
            common_names TEXT,
            raptor_group TEXT,
            is_raptor BOOLEAN
);
    ''')

    # Create mammals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mammals (
            species_code TEXT PRIMARY KEY,
            "order" TEXT NOT NULL,
            family TEXT NOT NULL,
            scientific_name TEXT NOT NULL,
            common_names TEXT,
            predator_group TEXT,
            is_large_predator BOOLEAN
        );
    ''')


    # Create records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            park_code TEXT NOT NULL,
            species_code TEXT NOT NULL,
            record_status TEXT NOT NULL CHECK (record_status IN ('In Review', 'Approved')),
            occurrence TEXT NOT NULL CHECK (occurrence IN ('Not Present (False Report)', 'Not Present (Historical Report)', 'Not Present', 'Not Confirmed', 'Present')),
            nativeness TEXT NOT NULL CHECK (nativeness IN ('Not Native', 'Unknown', 'Native')),
            abundance TEXT NOT NULL CHECK (abundance IN ('Rare', 'Uncommon', 'Unknown', 'Occasional', 'Common', 'Abundant')),
            seasonality TEXT,
            conservation_status TEXT NOT NULL CHECK (conservation_status IN ('Least Concern', 'Species of Concern', 'In Recovery', 'Under Review', 'Threatened', 'Proposed Endangered', 'Endangered')),
            is_protected BOOLEAN,
            PRIMARY KEY (park_code, species_code)
        );
    ''')
    logger.info("SQLite schema created successfully.")

# Functions to clean and prepare parks data
def clean_park_name(name):
    """Clean park names by removing specific keywords."""
    return re.sub(r' National Parks?$| National Park and Preserve$', '', name, flags=re.IGNORECASE).strip()


def acres_to_sq_km(acres):
    """Convert acres to square kilometers."""
    return round(acres * 0.00404686, 2)


def prepare_parks_dataframe(df, logger):
    """Prepare parks DataFrame by cleaning park names and converting acres."""
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df['square_km'] = df['acres'].apply(acres_to_sq_km)
    df['park_name'] = df['park_name'].apply(clean_park_name)
    logger.info("Parks DataFrame prepared with square kilometers and cleaned park names.")
    return df

# Load parks data from CSV, clean it, and insert into SQLite
def load_parks_data(conn, csv_url, logger):
    """Load, clean, and insert parks data from CSV into SQLite."""
    response = requests.get(csv_url)
    df_parks = pd.read_csv(StringIO(response.text))
    df_parks = prepare_parks_dataframe(df_parks, logger)
    df_parks[['park_code', 'park_name', 'state', 'square_km', 'latitude', 'longitude']].to_sql('parks', conn,
                                                                                               if_exists='replace',
                                                                                               index=False)
    logger.info("Parks data loaded into SQLite.")


# Load species and records data from pickle files and insert into SQLite
def load_species_and_records_data(conn, logger):
    """Load species and records data from pickle files into SQLite."""
    # Load species data
    with open('path_to_species_pickle.pkl', 'rb') as f:
        species_data = pickle.load(f)
    df_species = pd.DataFrame(species_data)
    df_species.to_sql('species', conn, if_exists='replace', index=False)
    logger.info("Species data loaded into SQLite.")

    # Load records data
    with open('path_to_records_pickle.pkl', 'rb') as f:
        records_data = pickle.load(f)
    df_records = pd.DataFrame(records_data)
    df_records.to_sql('records', conn, if_exists='replace', index=False)
    logger.info("Records data loaded into SQLite.")


# Main function to run the process
def main():
    check_files_exist(required_files, logger)
    conn = sqlite3.connect(sqlite_db_path)
    create_sqlite_schema(conn)

    load_parks_data(conn, csv_url, logger)

    birds_master = pd.read_pickle(birds_file)
    load_data_to_sqlite(conn, birds_master, 'birds', logger)

    mammals_master = pd.read_pickle(mammals_file)
    load_data_to_sqlite(conn, mammals_master, 'mammals', logger)

    records_master = pd.read_pickle(records_file)
    load_data_to_sqlite(conn, records_master, 'records', logger)

    # Close the connection
    conn.close()
    logger.info("SQLite database creation and data loading completed successfully.")


if __name__ == "__main__":
    main()