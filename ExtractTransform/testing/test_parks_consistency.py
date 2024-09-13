import unittest
import pandas as pd
import geopandas as gpd


class TestConsistency(unittest.TestCase):

    def setUp(self):
        # Load records.csv
        self.records = pd.read_pickle('Data/record_master.pkl')

        # Load parks_points.geojson
        self.parks_points = gpd.read_file('Data/parks_points.geojson')

        # Load parks_shapes.geojson
        self.parks_shapes = gpd.read_file('Data/parks_shapes.geojson')

    def test_park_code_consistency(self):
        # Extract park_code from each file as sets
        records_codes = set(self.records['park_code'])
        points_codes = set(self.parks_points['park_code'])
        shapes_codes = set(self.parks_shapes['park_code'])

        # Check for consistency between records_master and parks_points.geojson
        if records_codes != points_codes:
            missing_in_points = records_codes - points_codes
            extra_in_points = points_codes - records_codes
            if missing_in_points:
                print(f"Missing in parks_points.geojson but in records_master: {missing_in_points}")
            if extra_in_points:
                print(f"Extra in parks_points.geojson not in records_master: {extra_in_points}")
            self.assertEqual(records_codes, points_codes, "Mismatch between records.csv and parks_points.geojson")

        # Check for consistency between records_master and parks_shapes.geojson
        if records_codes != shapes_codes:
            missing_in_shapes = records_codes - shapes_codes
            extra_in_shapes = shapes_codes - records_codes
            if missing_in_shapes:
                print(f"Missing in parks_shapes.geojson but in records_master: {missing_in_shapes}")
            if extra_in_shapes:
                print(f"Extra in parks_shapes.geojson not in records_master: {extra_in_shapes}")
            self.assertEqual(records_codes, shapes_codes, "Mismatch between records.csv and parks_shapes.geojson")

        # Check for consistency between parks_points.geojson and parks_shapes.geojson
        if points_codes != shapes_codes:
            missing_in_shapes_from_points = points_codes - shapes_codes
            extra_in_shapes_from_points = shapes_codes - points_codes
            if missing_in_shapes_from_points:
                print(f"Missing in parks_shapes.geojson but in parks_points.geojson: {missing_in_shapes_from_points}")
            if extra_in_shapes_from_points:
                print(f"Extra in parks_shapes.geojson not in parks_points.geojson: {extra_in_shapes_from_points}")
            self.assertEqual(points_codes, shapes_codes, "Mismatch between parks_points.geojson and parks_shapes.geojson")

if __name__ == '__main__':
    unittest.main()