# US National Parks; A Project In Data Engineering

----

## Overview

This project aims to create a partially normalized OLAP (Online Analytical Processing) database to support complex analytical queries about the U.S. National Parks, with a specific focus on avian biodiversity. 

Leveraging public domain datasets provided by the U.S. National Parks Service, this project integrates species and geospatial data, transforming it into a structure optimized for analytical queries.

**Key Objectives:**
- Normalize and wrangle a large categorical dataset, addressing inconsistencies, duplicates, and data quality issues.
- Integrate geospatial data to enable spatial analysis of relationships between species and national park locations.
- Create an OLAP database schema optimized for performance and complex querying.
- Develop a robust ETL (Extract, Transform, Load) pipeline, incorporating unit testing and logging for data integrity and reproducibility.

----

## ETL Pipeline

The ETL pipeline consists of multiple stages, each responsible for extracting, transforming, and loading specific data related to U.S. National Parks and their biodiversity.

### [`extract_transform_parks.py`](extract_transform_parks.py) 

**Deliverables:** [`parks_points.geojson`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/parks_points.geojson) & [`parks_shapes.geojson`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/parks_shapes.geojson)

- **Data Source:** 
  - Geospatial data was sourced from [Data.gov](https://catalog.data.gov/dataset/national-park-boundaries/resource/cee04cfe-f439-4a65-91c0-ca2199fa5f93), an official site of the US government.
  - National Park data was sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/nationalparkservice/park-biodiversity?select=parks.csv).
- **Data Transformation:**
  - Normalized National Park codes using regex and fuzzy pattern matching to handle discrepancies between datasets.
  - Created two GeoJSON files with POINT and POLYGON geometries to represent park locations and boundaries.
  - Fields include: `park_code`, `park_name`, `state`, `square_km`, and `geometry`.
  
Spatial data is plotted below on a basemap to visualize park locations and boundaries.

![USParksShapes.png](Images/USParksShapes.png)

----

### [`extract_transform_species.py`](extract_transform_species.py)

**Deliverables:** [`birds.pkl`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/birds.pkl) & [`birds.csv`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/birds.csv)

- **Data Source:** Species data was sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/nationalparkservice/park-biodiversity?select=species.csv).
- **Data Transformation:**
  - Cleaned and standardized species information, focusing on avian species.
  - Managed categorical data by defining orders for categories such as conservation status, abundance, and nativeness.
  - Handled missing values using domain-specific fill values and adjusted data types to optimize for storage and analysis.
  - Applied fuzzy matching to resolve ambiguities in species names.

----

### [`extract_transform_records.py`](extract_transform_records.py)

**Deliverables:** [`species.csv`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/species.csv) & [`records.csv`](https://github.com/pineapple-bois/USNationalParks/blob/main/DATA/records.csv)

- **Data Integration:** 
  - Merged species data with park data to create a comprehensive dataset linking species to specific parks.
  - Generated unique identifiers for species and parks to maintain referential integrity.
  - Structured the data to fit into a schema optimized for OLAP queries.
  
- **Data Quality Assurance:**
  - Conducted thorough data validation, including unit tests to ensure data consistency and correctness.
  - Implemented logging to capture the data transformation process, including any anomalies or issues detected during ETL.

----

### Directory Structure

```markdown
USNationalParks/
│
├── DATA/
│   ├── Backups/
│   ├── Masters/
│   │   ├── parks.csv
│   │   └── species.csv
│   ├── birds.csv
│   ├── birds.pkl
│   ├── nps_boundary.geojson
│   ├── nps_boundary.xml
│   ├── parks_points.geojson
│   ├── parks_shapes.geojson
│   ├── records.csv
│   └── species.csv
│
├── Functions/
│   ├── __init__.py
│   └── transformation_functions.py
│
├── Images/
│   ├── USParksLatLong.png
│   └── USParksShapes.png
│
├── Tests/
│   ├── __init__.py
│   ├── test_parks_consistency.py
│   └── test_record_consistency.py
│
├── extract_transform_parks.py
├── extract_transform_records.py
├── extract_transform_species.py
│
└── README.md
```

----

## Work in Progress

### Data Normalization and Cleaning
- **Species Data**: Extracted and partially normalized species data with a focus on avian species, handling inconsistencies and duplicates.
- **Park Boundaries**: Integrated GeoJSON data for US National Parks, enabling spatial analysis capabilities within the OLAP database.
- **Handling Missing Values**: Strategically filled missing values using domain-specific defaults (e.g., 'Least Concern' for conservation status).

### Unit Testing and Validation
- Developed unit tests to validate data transformations, ensuring expected dimensions, data types, and categorical orders.
- Implemented detailed logging to track data integrity and transformation steps.

### Logging and Error Handling
- Configured logging to document transformation processes, including information on unique values, data types, and detected anomalies.
- Error handling mechanisms are in place to capture and log unexpected behavior during data processing.

----

## Next Steps

### Database Schema Design
- Define a normalized schema for the PostgreSQL database, incorporating both spatial and non-spatial data elements.
- Design enums and constraints to enforce data integrity and optimize query performance.

### Data Loading
- Implement data loading scripts to populate the PostgreSQL database with transformed data.
- Utilize SQL scripts or an ORM (e.g., SQLAlchemy) to manage database interactions.

### Advanced Analysis and Reporting
- Develop OLAP cubes and other analytical structures to support complex querying and reporting.


----

