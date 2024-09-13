# Biodiversity in US National Parks; An ETL Pipeline

----

## Overview

This project aims to create a partially normalized OLAP (Online Analytical Processing) database to support complex analytical queries about the U.S. National Parks, with a specific focus on biodiversity. 

Leveraging public domain datasets provided by the U.S. National Parks Service, this project integrates species and geospatial data, transforming it into a structure optimized for analytical queries.

**Key Objectives:**
- Normalize and wrangle a large categorical dataset, addressing inconsistencies, duplicates, and data quality issues.
- Integrate geospatial data to enable spatial analysis of relationships between species and national park locations.
- Create an OLAP database schema optimized for performance and complex querying.
- Develop a robust ETL (Extract, Transform, Load) pipeline, incorporating unit testing and logging for data integrity and reproducibility.


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

