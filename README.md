# Biodiversity in US National Parks: An ETL Pipeline

---

### Overview

This project builds a partially normalised OLAP (Online Analytical Processing) database to support complex analytical queries on U.S. National Parks, focusing on their biodiversity. The ETL pipeline processes raw biodiversity data from the U.S. National Parks Service into a structured format for analysis in a PostgreSQL database.

### Pipeline includes:

- **Data Extraction**: Collects raw data from public sources.
- **Data Transformation**: Cleans and standardises the data, fixing missing values and normalising species names.
- **Data Loading**: Loads the transformed data into a PostgreSQL database for complex analysis.

### Features

- **Data Quality**: Resolves duplicates, inconsistencies, and missing values to ensure reliable data.
- **Geospatial Analysis**: Integrates geospatial data to explore species distributions within park boundaries.
- **Modular Design**: Uses an object-oriented approach, making it easy to add new species categories.
- **Error Handling and Logging**: Provides detailed logs and manages errors efficiently.
- **Consistency Checks**: Ensures unique constraints for primary keys (`park_code` and `species_code`).

---

### [Extract Transform Module](ExtractTransform)

Data was sourced from an open-source [Kaggle Dataset](https://www.kaggle.com/datasets/nationalparkservice/park-biodiversity?select=species.csv) provided by the U.S. National Parks Service.

The main transformation process subsets the data by category (`Bird`, `Mammal`, `Reptile`, etc.) and normalises the taxonomic records. Species often have multiple common names, typographical errors, and ambiguities. To address this, a master table of taxonomic information (`order`, `family`, `scientific_name`, `common_name`) was created, assigning a unique reference number to each species.

Example: Considering *Accipiter gentilis* (Northern Goshawk), various common names were normalised:

| scientific_name    | common_names                               |
|--------------------|--------------------------------------------|
| Accipiter gentilis | Eastern Goshawk, Goshawk, Northern Goshawk |
| Accipiter gentilis | Northern Goshawk                           |
| Accipiter gentilis | Goshawk                                    |

The standardisation process used regular expressions, `collections.Counter` objects, and cross-referencing to resolve ambiguities and errors, improving the quality and consistency of the data.

#### Object-Oriented Approach

The abstract base class model allows for category-specific transformation strategies. For instance:

- **Birds**: The `BirdTransformStrategy` handles subspecies and identifies attributes like `raptor_group` and `is_raptor`.
- **Plants**: Represents about 65,000 records, often including subspecies and hybrids. The object orientated structure supports the development of plant-specific filters and attributes, such as identifying `cacti` or `invasive_species`.

---

### Very Quick Start Guide

This guide will help you explore the SQLite version of the database `national_parks_lite.db` using [SQLAlchemy](https://www.sqlalchemy.org) and Pandas. It was created from [SQL/sqlite_db.py](SQL/sqlite_db.py) 

#### Prerequisites

You can install the necessary packages using:

```bash
pip install sqlalchemy pandas
```

1.	Create a connection to the SQLite database using SQLAlchemy:

```python
from sqlalchemy import create_engine, inspect
import pandas as pd

# Create DB engine
engine = create_engine('sqlite:///national_parks_lite.db')
print(f"Engine type: {type(engine)}")
```

2. Use SQLAlchemy’s inspect function to list all the tables and explore their columns:

```python
# Use the inspect function to get table names
inspector = inspect(engine)
tables = inspector.get_table_names()
print(f"Tables:\n{tables}")

# Explore columns in each table
for table in tables:
    columns = inspector.get_columns(table)
    print(f"\nTable: {table}")
    for column in columns:
        print(f"  {column['name']} - {column['type']}")
```

3.	Run a Sample Query:
- SQL queries are formatted as multi-line strings in Python
```python
# list the top 10 National parks by bird of prey species
sql_query = '''
SELECT
    p.park_name,
    p.state,
    COUNT(*) AS raptor_count
FROM records AS r
LEFT JOIN birds AS b ON r.species_code = b.species_code
LEFT JOIN parks AS p ON r.park_code = p.park_code
WHERE b.is_raptor = True
GROUP BY p.park_name, p.state
ORDER BY raptor_count DESC
LIMIT 10;
'''
```

```python
# Identify National Parks with a population of 'California Condor'
sql_query2 = '''
SELECT
    r.park_code AS park_code,
    p.park_name,
    p.state,
    b.scientific_name,
    b.common_names,
    r.occurrence,
    r.nativeness,
    r.abundance,
    r.seasonality,
    r.conservation_status
FROM records AS r
LEFT JOIN birds AS b ON r.species_code = b.species_code
LEFT JOIN parks AS p ON r.park_code = p.park_code
WHERE b.common_names = 'California Condor'
ORDER BY r.occurrence DESC, r.abundance DESC;
'''
```

4.	Load the query results into Pandas DataFrames facilitating Exploratory Data Analysis within a Jupyter Notebook
```python
df = pd.read_sql_query(sql_query, engine)
df2 = pd.read_sql_query(sql_query2, engine)
```

---

### PostgreSQL Database Design

The PostgreSQL database [schema](schema.sql) is designed to manage U.S. National Parks data efficiently. It includes custom ENUM types, primary keys, foreign keys, and a composite primary key to maintain data integrity and support complex analytical queries.

#### Spatial Data Integration

The schema integrates geospatial data using [PostGIS](https://postgis.net), with `park_points` and `park_shapes` tables storing park locations as points and boundaries as polygons. 

#### Entity Relationship Diagram

![img](Images/EntityRelationship.png)


#### Primary Keys

- Each table is uniquely identified by a primary key:
  - `parks` table: `park_code`
  - `park_points` and `park_shapes` tables: `park_code`, aligning spatial data with corresponding parks.
  - `birds` and `mammals` tables: `species_code`, ensuring unique identification of species.

- **Composite Primary Key**:
  - The `records` table uses a composite primary key (`park_code`, `species_code`), supporting a many-to-many relationship between parks and species.

#### Foreign Keys

- Foreign keys enforce relationships between tables:
  - `records` table: References `park_code` from the `parks` table, with `ON DELETE CASCADE` to remove related records when a park is deleted.
  - `park_points` and `park_shapes` tables: Also reference `park_code` from the `parks` table.

   
#### Custom ENUM types 

These standardise and constrain values for categorical fields in the `records` table, improving data quality and query efficiency:

- **`record_status`**: `'In Review'`, `'Approved'`
- **`occurrence_type`**: Values range from `'Not Present (False Report)'` to `'Present'`
- **`nativeness_type`**: `'Native'`, `'Not Native'`, `'Unknown'`
- **`abundance_type`**: Includes `'Rare'`, `'Common'`, `'Abundant'`, among others
- **`conservation_status_type`**: From `'Least Concern'` to `'Endangered'` as per the [International Union for the Conservation of Nature](https://www.iucnredlist.org)

----

### Getting Started Guide

#### 1. Create the dataset within your environment

Clone the Repository:
```bash
git clone https://github.com/your-repo-url
```

Create a virtual environment:
```bash
# Navigate into the project directory
cd your-repo-directory

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install Required Packages:
- Use the provided `requirements.txt` file to install necessary Python packages:
```bash
pip install -r requirements.txt
```

Extract and Transform the raw data:
- Execute the main script to run the extract and transform processes:
```bash
python data_extraction.py
 ```

The data extraction will create a new folder `Pipeline` in the current working directory which contains `BackupData`, `Logs`, `FinalData` and `Images`.

----

#### 2. Set up the PostgreSQL database

The database schema uses the PostGIS extension which has a great [Getting Started Guide](https://postgis.net/documentation/getting_started/) which covers PostgreSQL and PostGIS installation appropriate to your OS. 

Assuming you have PostgreSQL and PostGIS correctly installed and configured, follow these steps:

Create the Database:
```sql
-- Open the PostgreSQL command line or use a client like pgAdmin
CREATE DATABASE national_parks;
```
```sql
-- Connect to the national_parks database
\c national_parks
```

----

#### 3. Run the Schema

Run the provided schema file to set up the tables:
```bash
# From the root directory of the repository, execute:
psql -d national_parks -f schema.sql
```
---

#### 4. Populate the Tables

To populate the tables, use the [data_loading.py](data_loading.py) script, which will load the transformed data into your PostgreSQL database. Before running the script, ensure that you update your database username and password in the script:
1.	Open the data_loading.py file.
2. Update the following lines with your PostgreSQL credentials:
```python
db_user = 'YOUR_USER_NAME'       # Replace with your database username
db_password = 'YOUR_PASSWORD'    # Replace with your database password
```
3.	The script includes a function clear_table_data(engine, logger) that clears existing data from the tables if they already exist, ensuring that you start with a clean slate each time you run the script.
4. To execute the data loading process, run:
```bash
# Execute the data loading script
python data_loading.py
```
This script will:

- Clear existing data from the tables (if any).
- Load data from the FinalData directory into the respective tables (parks, birds, mammals, records, park_points, and park_shapes).
- Log the status of each operation to data_loading.log, including verification of the data insertion process.

Ensure the script runs without errors to fully populate your database with the extracted and transformed data, ready for analysis!

---

### Future Enhancements

- **Extend to Additional Species**: Expand the pipeline to include more species categories, such as reptiles, amphibians, and plants, leveraging the modular design of the ETL process.
- **Enhanced Spatial Analysis**: Integrate more advanced spatial analysis tools and visualisations to explore relationships between species distributions and park geographies.
- **Dynamic Taxonomic Updates**: Connect the database to a reputable taxonomic data source, such as the [Integrated Taxonomic Information System (ITIS)](https://www.itis.gov/) or [Catalogue of Life](https://www.catalogueoflife.org/), to dynamically update species information in response to taxonomic changes. This will ensure that the database remains current with the latest scientific consensus, allowing for automatic updates to species records when new classifications or name changes occur.

The [Northern Goshawk](https://en.wikipedia.org/wiki/Northern_goshawk) was designated as a separate species *Astur atricapillus*, the "American Goshawk" in 2023. A master table of taxonomic information makes it easy to update the database if and when new science on species formerly considered conspecific to others comes to light. Integrating a connection to live taxonomic databases would further enhance the robustness and accuracy of the system, keeping it aligned with current taxonomy.

---

### Contributions

We welcome issues, feedback, and contributions from the community to improve this project. If you encounter any issues, have suggestions for enhancements, or wish to contribute, please feel free to raise an issue on our GitHub repository or contact us directly.

For contributions, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes, ensuring code quality and consistency with the project's guidelines.
3. Submit a pull request with a detailed description of your changes.

For any inquiries or further discussion, contact us via email at: [iwood@posteo.net](mailto:iwood@posteo.net)

We look forward to your contributions!

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE.md) [![Pineapple Bois](https://img.shields.io/badge/Website-Pineapple_Bois-5087B2.svg?style=flat&logo=telegram)](https://pineapple-bois.github.io)

----
