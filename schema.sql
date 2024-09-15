-- Enable PostGIS extension for handling spatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create ENUM types for categorical fields
CREATE TYPE record_status AS ENUM ('In Review', 'Approved');
CREATE TYPE occurrence_type AS ENUM ('Not Present (False Report)', 'Not Present (Historical Report)', 'Not Present', 'Not Confirmed', 'Present');
CREATE TYPE nativeness_type AS ENUM ('Not Native', 'Unknown', 'Native');
CREATE TYPE abundance_type AS ENUM ('Rare', 'Uncommon', 'Unknown', 'Occasional', 'Common', 'Abundant');
CREATE TYPE conservation_status_type AS ENUM ('Least Concern', 'Species of Concern', 'In Recovery', 'Under Review', 'Threatened', 'Proposed Endangered', 'Endangered');

-- Create birds table
CREATE TABLE birds (
    species_code VARCHAR PRIMARY KEY,
    "order" VARCHAR NOT NULL,
    family VARCHAR NOT NULL,
    scientific_name VARCHAR NOT NULL,
    common_names VARCHAR,
    raptor_group VARCHAR,
    is_raptor BOOLEAN
);

-- Create mammals table
CREATE TABLE mammals (
    species_code VARCHAR PRIMARY KEY,
    "order" VARCHAR NOT NULL,
    family VARCHAR NOT NULL,
    scientific_name VARCHAR NOT NULL,
    common_names VARCHAR,
    predator_group VARCHAR,
    is_large_predator BOOLEAN
);

-- Create records table
CREATE TABLE records (
    park_code VARCHAR NOT NULL,
    species_code VARCHAR NOT NULL,
    record_status record_status NOT NULL,
    occurrence occurrence_type NOT NULL,
    nativeness nativeness_type NOT NULL,
    abundance abundance_type NOT NULL,
    seasonality VARCHAR,
    conservation_status conservation_status_type NOT NULL,
    is_protected BOOLEAN,
    PRIMARY KEY (park_code, species_code)
);

CREATE TABLE parks (
    park_code VARCHAR PRIMARY KEY,
    park_name VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    square_km NUMERIC
);

-- Create park_points table
CREATE TABLE park_points (
    park_code VARCHAR PRIMARY KEY,
    geometry GEOMETRY(Point, 4326),
    FOREIGN KEY (park_code) REFERENCES parks(park_code) ON DELETE CASCADE
);

-- Create park_shapes table
CREATE TABLE park_shapes (
    park_code VARCHAR PRIMARY KEY,
    geometry GEOMETRY(Geometry, 4326),
    FOREIGN KEY (park_code) REFERENCES parks(park_code) ON DELETE CASCADE
);