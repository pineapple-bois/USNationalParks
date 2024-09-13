import pandas as pd

from ExtractTransform.extract_transform_parks import process_parks_data
from ExtractTransform.extract_species import ExtractSpecies
from ExtractTransform.transform_species import TransformSpecies
from ExtractTransform.transform_records import TransformRecords


if __name__ == "__main__":
    # Step 1: Extract the parks data
    parks_points, parks_shapes = process_parks_data(create_plots=True)

    # Step 2: Extract the master species data
    species_data = ExtractSpecies()
    species_df = species_data.dataframe

    # Step 3: Transform the species data for each category
    bird_data = TransformSpecies("Bird", dataframe=species_df)
    bird_df = bird_data.dataframe

    mammal_data = TransformSpecies("Mammal", dataframe=species_df)
    mammal_df = mammal_data.dataframe

    # reptile_data = TransformSpecies("Reptile", dataframe=species_df)
    # reptile_df = reptile_data.dataframe

    # Step 4: Generate Database table data
    all_records = TransformRecords([bird_df, mammal_df])
    birds_master = all_records.bird_records
    mammals_master = all_records.mammal_records
    records_master = all_records.all_records
