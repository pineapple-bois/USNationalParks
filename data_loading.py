import os
import sys
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import logging

from ExtractTransform.utils import DataFrameUtils


def main():
    # Database connection parameters
    db_user = 'YOUR_USERNAME'
    db_password = 'YOUR_PASSWORD'
    db_host = 'localhost'
    db_port = '5432'  # Usually 5432
    db_name = 'national_parks'   # Change if you created Db with another name

    db_connection_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_connection_url)

    # Setup Logger
    logger = DataFrameUtils.setup_logger('insertion_logger', 'data_loading.log')
    logger.info('Starting data extraction and transformation pipeline.')

    # Clear existing data
    logger.info('Clearing existing data from tables.')
    clear_table_data(engine, logger)

    try:
        # Define file paths
        data_dir = 'Pipeline/FinalData'
        birds_file = os.path.join(data_dir, 'bird_master.pkl')
        mammals_file = os.path.join(data_dir, 'mammal_master.pkl')
        reptiles_file = os.path.join(data_dir, 'reptile_master.pkl')
        records_file = os.path.join(data_dir, 'record_master.pkl')
        parks_points_file = os.path.join(data_dir, 'parks_points.geojson')
        parks_shapes_file = os.path.join(data_dir, 'parks_shapes.geojson')

        # Check if all required files exist
        required_files = [birds_file, mammals_file, reptiles_file, records_file, parks_points_file, parks_shapes_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            logger.error(f"Missing required data files: {missing_files}")
            print("Data files are missing. Please run the data extraction step before loading data into the database.")
            sys.exit(1)  # Exit the script with an error code

        # Load DataFrames from files
        logger.info('Loading DataFrames from files.')
        birds_master = pd.read_pickle(birds_file)
        mammals_master = pd.read_pickle(mammals_file)
        reptiles_master = pd.read_pickle(reptiles_file)
        records_master = pd.read_pickle(records_file)
        parks_points = gpd.read_file(parks_points_file)
        parks_shapes = gpd.read_file(parks_shapes_file)

        # Prepare DataFrames for insertion
        logger.info('Preparing DataFrames for database insertion.')

        # Prepare parks data
        parks = pd.DataFrame(parks_points[['park_code', 'park_name', 'state', 'square_km']].drop_duplicates())

        # Prepare geospatial data
        parks_points = parks_points[['park_code', 'geometry']].drop_duplicates()
        parks_shapes = parks_shapes[['park_code', 'geometry']].drop_duplicates()

        # Ensure CRS and validate geometries
        parks_points = parks_points.set_crs(epsg=4326)
        parks_shapes = parks_shapes.set_crs(epsg=4326)
        parks_shapes['geometry'] = parks_shapes['geometry'].apply(
            lambda geom: geom.buffer(0) if not geom.is_valid else geom
        )

        # Insert data into the database
        logger.info('Inserting data into the database.')
        with engine.begin() as connection:
            # Insert parks data
            parks.to_sql('parks', connection, if_exists='append', index=False)
            logger.info('Inserted parks data.')

            # Insert birds data
            birds_master.to_sql('birds', connection, if_exists='append', index=False)
            logger.info('Inserted birds data.')

            # Insert mammals data
            mammals_master.to_sql('mammals', connection, if_exists='append', index=False)
            logger.info('Inserted mammals data.')

            # Insert reptiles data
            reptiles_master.to_sql('reptiles', connection, if_exists='append', index=False)
            logger.info('Inserted reptiles data.')

            # Insert records data
            records_master.to_sql('records', connection, if_exists='append', index=False)
            logger.info('Inserted records data.')

            # Insert geospatial data
            parks_points.to_postgis('park_points', connection, if_exists='append', index=False)
            logger.info('Inserted park_points data.')

            parks_shapes.to_postgis('park_shapes', connection, if_exists='append', index=False)
            logger.info('Inserted park_shapes data.')

        # Verification step
        logger.info('Verifying data insertion.')
        verify_data_insertion(engine, logger)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

    logger.info('Data loading process completed successfully.')

def verify_data_insertion(engine, logger):
    """Verifies that data has been inserted into the database."""
    with engine.connect() as conn:
        # Verify counts in each table
        tables = ['parks', 'birds', 'mammals', 'records', 'park_points', 'park_shapes']
        for table in tables:
            result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
            count = result.scalar()
            logger.info(f"Table '{table}' has {count} records.")

def clear_table_data(engine, logger):
    """Clears data from specified tables in the database if it already exists."""
    tables = ['parks', 'birds', 'mammals', 'records', 'park_points', 'park_shapes']
    with engine.connect() as conn:
        # Disable foreign key checks (for PostgreSQL)
        conn.execute(text('SET session_replication_role = replica;'))
        try:
            for table in tables:
                # Truncate the table to remove all rows
                conn.execute(text(f'TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;'))
                logger.info(f"Cleared data from table '{table}'.")
            # Re-enable foreign key checks
            conn.execute(text('SET session_replication_role = DEFAULT;'))
        except Exception as e:
            # Re-enable foreign key checks if an error occurs
            conn.execute(text('SET session_replication_role = DEFAULT;'))
            logger.error(f"Failed to clear data from table: {e}")
            raise
        conn.commit()  # Ensure all changes are committed
    logger.info('All specified tables have been cleared.')

if __name__ == '__main__':
    main()